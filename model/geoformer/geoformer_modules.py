from collections import OrderedDict
import spconv
import torch
import torch.nn as nn
from model.transformer import TransformerEncoder
from spconv.modules import SparseModule
from util.warpper import BatchNorm1d, Conv1d


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, use_backbone_transformer=False, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        blocks = {
            "block{}".format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key="subm{}".format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)
        if len(nPlanes) <= 2 and use_backbone_transformer:
            d_model = 128
            self.before_transformer_linear = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=4, d_ff=64)
            self.after_transformer_linear = nn.Linear(d_model, nPlanes[0])
        else:
            self.before_transformer_linear = None
            self.transformer = None
            self.after_transformer_linear = None
        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, use_backbone_transformer, indice_key_id=indice_key_id + 1
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key="spconv{}".format(indice_key_id)
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key="subm{}".format(indice_key_id)
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        if self.before_transformer_linear:
            batch_ids = output.indices[:, 0]
            xyz = output.indices[:, 1:].float()
            feats = output.features
            before_params_feats = self.before_transformer_linear(feats)
            feats = self.transformer(xyz=xyz, features=before_params_feats, batch_ids=batch_ids)
            feats = self.after_transformer_linear(feats)
            output.features = feats

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
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=(norm is None)
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
