from collections import OrderedDict
import spconv
import torch
import torch.nn as nn
from model.transformer import TransformerEncoder
from spconv.pytorch.conv import SubMConv3d, SparseInverseConv3d, SparseConv3d
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.core import SparseConvTensor
from util.warpper import BatchNorm1d, Conv1d
import numpy as np


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = SparseSequential(nn.Identity())
        else:
            self.i_branch = SparseSequential(
                SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
            SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        identity = SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )

        output = self.conv_branch(input)
        new_features = output.features + self.i_branch(identity).features
        output = output.replace_feature(new_features)
        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(
        self,
        nPlanes,
        norm_fn,
        block_reps,
        block,
        use_backbone_transformer=False,
        indice_key_id=1,
    ):

        super().__init__()

        self.nPlanes = nPlanes
        blocks = {
            "block{}".format(i): block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key="subm{}".format(indice_key_id),
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = SparseSequential(blocks)
        if len(nPlanes) <= 2 and use_backbone_transformer:
            d_model = 128
            self.before_transformer_linear = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(
                d_model=d_model, N=2, heads=4, d_ff=64
            )
            self.after_transformer_linear = nn.Linear(d_model, nPlanes[0])
        else:
            self.before_transformer_linear = None
            self.transformer = None
            self.after_transformer_linear = None
        if len(nPlanes) > 1:
            self.conv = SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            self.u = UBlock(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                use_backbone_transformer,
                indice_key_id=indice_key_id + 1,
            )

            self.deconv = SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key="subm{}".format(indice_key_id),
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1)
            )

            output = self.blocks_tail(output)

        if self.before_transformer_linear:
            batch_ids = output.indices[:, 0]
            xyz = output.indices[:, 1:].float()
            feats = output.features
            before_params_feats = self.before_transformer_linear(feats)
            feats = self.transformer(
                xyz=xyz, features=before_params_feats, batch_ids=batch_ids
            )
            feats = self.after_transformer_linear(feats)
            output = output.replace_feature(feats)

        return output


def conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False):
    def make_conv(in_channels, out_channels):
        conv_func = Conv1d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=(norm is None),
        )

        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)

        module = [
            conv,
        ]
        if norm is not None and len(norm) > 0:
            norm_module = BatchNorm1d(out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


def random_downsample(batch_offsets, batch_size, n_subsample=30000):
    idxs_subsample = []
    idxs_subsample_raw = []
    for b in range(batch_size):
        start, end = batch_offsets[b], batch_offsets[b + 1]
        num_points_b = (end - start).cpu()

        if n_subsample == -1 or n_subsample >= num_points_b:
            new_inds = torch.arange(
                num_points_b, dtype=torch.long, device=batch_offsets.device
            )
        else:
            new_inds = torch.tensor(
                np.random.choice(num_points_b, n_subsample, replace=False),
                dtype=torch.long,
                device=batch_offsets.device,
            )
        idxs_subsample_raw.append(new_inds)
        idxs_subsample.append(new_inds + start)
    idxs_subsample = torch.cat(idxs_subsample)  # N_subsample: batch x 20000
    # idxs_subsample_raw = torch.cat(idxs_subsample_raw)
    return idxs_subsample, idxs_subsample_raw
