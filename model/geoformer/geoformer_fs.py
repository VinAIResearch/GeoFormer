import functools
import spconv as spconv
import torch
import torch.nn as nn


import faiss  # make faiss available
import faiss.contrib.torch_utils
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotesSeparate
from model.geoformer.geodesic_utils import cal_geodesic_vectorize
from model.geoformer.geoformer_modules import ResidualBlock, UBlock, conv_with_kaiming_uniform
from model.helper import GenericMLP
from model.pos_embedding import PositionEmbeddingCoordsSine
from model.transformer_detr import TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F
from util import utils
from util.config import cfg


class GeoFormerFS(nn.Module):
    def __init__(self):
        super().__init__()

        input_c = cfg.input_channel
        if cfg.use_coords:
            input_c += 3

        m = cfg.m

        classes = cfg.classes

        self.prepare_epochs = cfg.prepare_epochs

        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )
        self.unet = UBlock(
            [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
            norm_fn,
            2,
            ResidualBlock,
            use_backbone_transformer=True,
            indice_key_id=1,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        # semantic segmentation
        self.semantic = nn.Sequential(
            nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU(), nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU()
        )
        self.semantic_linear = nn.Linear(m, classes, bias=True)

        # for instance embedding
        self.output_dim = m
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        mask_tower = []
        for i in range(self.mask_conv_num):
            mask_tower.append(conv_block(m, m))
        mask_tower.append(nn.Conv1d(m, self.output_dim, 1))
        self.add_module("mask_tower", nn.Sequential(*mask_tower))

        # convolution before the condinst take place (convolution num before the generated parameters take place)
        before_embedding_conv_num = 1
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        before_embedding_tower = []
        for i in range(before_embedding_conv_num - 1):
            before_embedding_tower.append(conv_block(cfg.dec_dim, cfg.dec_dim))
        before_embedding_tower.append(conv_block(cfg.dec_dim, self.output_dim))
        self.add_module("before_embedding_tower", nn.Sequential(*before_embedding_tower))

        # cond inst generate parameters for
        self.use_coords = True
        self.embedding_conv_num = 2
        weight_nums = []
        bias_nums = []
        for i in range(self.embedding_conv_num):
            if i == 0:
                if self.use_coords:
                    weight_nums.append((self.output_dim + 3) * self.output_dim)
                else:
                    weight_nums.append(self.output_dim * self.output_dim)
                bias_nums.append(self.output_dim)
            elif i == self.embedding_conv_num - 1:
                weight_nums.append(self.output_dim)
                bias_nums.append(1)
            else:
                weight_nums.append(self.output_dim * self.output_dim)
                bias_nums.append(self.output_dim)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Conv1d(self.output_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        """ Set aggregate """
        set_aggregate_dim_out = 2 * m
        mlp_dims = [m, 2 * m, 2 * m, set_aggregate_dim_out]
        self.set_aggregator = PointnetSAModuleVotesSeparate(
            radius=0.2,
            nsample=64,
            npoint=cfg.n_decode_point,
            mlp=mlp_dims,
            normalize_xyz=True,
        )

        """ Position embedding """
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=cfg.dec_dim, pos_type="fourier", normalize=True)

        """ DETR-Decoder """
        decoder_layer = TransformerDecoderLayer(
            d_model=cfg.dec_dim,
            nhead=cfg.dec_nhead,
            dim_feedforward=cfg.dec_ffn_dim,
            dropout=cfg.dec_dropout,
            normalize_before=True,
            use_rel=True,
        )

        self.decoder = TransformerDecoder(decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True)

        self.query_projection = GenericMLP(
            input_dim=cfg.dec_dim,
            hidden_dims=[cfg.dec_dim],
            output_dim=cfg.dec_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=set_aggregate_dim_out * 3,
            hidden_dims=[set_aggregate_dim_out * 3],
            output_dim=cfg.dec_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        self.similarity_net = nn.Sequential(
            nn.Linear(3 * set_aggregate_dim_out, 3 * set_aggregate_dim_out, bias=True),
            norm_fn(3 * set_aggregate_dim_out),
            nn.ReLU(),
            nn.Linear(3 * set_aggregate_dim_out, 3 * set_aggregate_dim_out, bias=True),
            norm_fn(3 * set_aggregate_dim_out),
            nn.ReLU(),
            nn.Linear(3 * set_aggregate_dim_out, 1, bias=True),
        )

        self.init_knn()

        self.apply(self.set_bn_init)

        for mod_name in self.fix_module:
            mod = getattr(self, mod_name)
            for param in mod.parameters():
                param.requires_grad = False

    def init_knn(self):
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = True
        faiss_cfg.device = 0

        self.geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss_cfg)

    def train(self, mode=True):
        super().train(mode)
        for mod_name in self.fix_module:
            mod = getattr(self, mod_name)
            for m in mod.modules():
                m.eval()

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm1d") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def generate_proposal(
        self,
        mask_logits,
        similarity_score,
        fg_idxs,
        batch_offsets,
        logit_thresh=0.5,
        score_thresh=0.5,
        npoint_thresh=100,
        sim_score_thresh=0.5,
    ):

        # NOTE only batch 1 when test
        b = 0
        start = batch_offsets[b]
        end = batch_offsets[b + 1]
        num_points = end - start

        mask_logit_b = mask_logits[b].sigmoid()
        similarity_score_b = similarity_score[b]

        similarity_score_cond = similarity_score_b >= sim_score_thresh
        mask_logit_b_bool = mask_logit_b >= logit_thresh

        proposals_npoints = torch.sum(mask_logit_b_bool, dim=1)
        npoints_cond = proposals_npoints >= npoint_thresh

        mask_logit_scores = torch.sum(mask_logit_b * mask_logit_b_bool.int(), dim=1) / (proposals_npoints + 1e-6)
        mask_logit_scores_cond = mask_logit_scores >= score_thresh

        scores = mask_logit_scores * torch.pow(similarity_score_b, 0.5)

        final_cond = similarity_score_cond & npoints_cond & mask_logit_scores_cond

        if torch.count_nonzero(final_cond) == 0:
            return [], []

        masks_final = mask_logit_b_bool[final_cond]
        scores_final = scores[final_cond]

        num_insts = scores_final.shape[0]
        proposals_pred = torch.zeros((num_insts, num_points), dtype=torch.int, device=mask_logit_b.device)

        inst_inds, point_inds = torch.nonzero(masks_final, as_tuple=True)

        point_inds = fg_idxs[point_inds]

        proposals_pred[inst_inds, point_inds] = 1
        return scores_final, proposals_pred

    def parse_dynamic_params(self, params, out_channels):
        assert params.dim() == 2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_instances * out_channels, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances * out_channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_instances, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances)

        return weight_splits, bias_splits

    def mask_heads_forward(
        self, geo_dist, mask_features, weights, biases, num_insts, coords_, fps_sampling_coords, use_geo=True
    ):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2, 1, 0).repeat(num_insts, 1, 1)  # num_inst * c * N_mask

        geo_dist = geo_dist.cuda()

        relative_coords = fps_sampling_coords[:, None, :] - coords_[None, :, :]  # N_inst * N_mask * 3

        if use_geo:
            n_queries, n_contexts = geo_dist.shape[:2]
            max_geo_dist_context = torch.max(geo_dist, dim=1)[0]  # n_queries
            max_geo_val = torch.max(max_geo_dist_context)
            max_geo_dist_context[max_geo_dist_context < 0] = max_geo_val
            max_geo_dist_context = torch.sqrt(max_geo_dist_context)

            max_geo_dist_context = max_geo_dist_context[:, None, None].expand(
                n_queries, n_contexts, 3
            )  # b x n_queries x n_contexts x 3

            cond = (geo_dist < 0).unsqueeze(-1).expand(n_queries, n_contexts, 3)
            relative_coords[cond] = relative_coords[cond] + max_geo_dist_context[cond] * torch.sign(
                relative_coords[cond]
            )

        relative_coords = relative_coords.permute(0, 2, 1)
        x = torch.cat([relative_coords, x], dim=1)  # num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask)  # 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x

    def get_mask_prediction(
        self, geo_dists, param_kernels, mask_features, locs_float_, fps_sampling_locs, batch_offsets_
    ):
        # param_kernels = param_kernels.permute(0, 2, 1, 3) # num_layers x batch x n_queries x channel
        num_layers, n_queries, batch = (
            param_kernels.shape[0],
            param_kernels.shape[1],
            param_kernels.shape[2]
        )

        outputs = []
        for l in range(num_layers):

            param_kernel = param_kernels[l]  # n_queries x batch x channel
            # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
            param_kernel2 = param_kernel.transpose(0, 1).flatten(0, 1)  # (batch * n_queries) * channel
            before_embedding_feature = self.before_embedding_tower(torch.unsqueeze(param_kernel2, dim=2))
            controllers = self.controller(before_embedding_feature).squeeze(dim=2)

            controllers = controllers.reshape(batch, n_queries, -1)

            mask_logits_list = []
            for b in range(batch):
                start = batch_offsets_[b]
                end = batch_offsets_[b + 1]

                if end - start == 0:
                    mask_logits_list.append(None)
                    continue

                controller = controllers[b]  # n_queries x channel
                weights, biases = self.parse_dynamic_params(controller, self.output_dim)

                mask_feature_b = mask_features[start:end, :]
                locs_float_b = locs_float_[start:end, :]
                fps_sampling_locs_b = fps_sampling_locs[b]

                geo_dist = geo_dists[b]

                mask_logits = self.mask_heads_forward(
                    geo_dist,
                    mask_feature_b,
                    weights,
                    biases,
                    n_queries,
                    locs_float_b,
                    fps_sampling_locs_b,
                    use_geo=True,
                )

                mask_logits = mask_logits.squeeze(dim=0)  # (n_queries) x N_mask
                mask_logits_list.append(mask_logits)

            output = {"mask_logits": mask_logits_list}
            outputs.append(output)
        return outputs

    def preprocess_input(self, batch_input, batch_size):
        voxel_coords = batch_input["voxel_locs"]  # (M, 1 + 3), long, cuda
        v2p_map = batch_input["v2p_map"]  # (M, 1 + maxActive), int, cuda
        locs_float = batch_input["locs_float"]  # (N, 3), float32, cuda
        feats = batch_input["feats"]  # (N, C), float32, cuda
        spatial_shape = batch_input["spatial_shape"]

        if cfg.use_coords:
            feats = torch.cat((feats, locs_float), 1).float()

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        sparse_input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        return sparse_input

    def process_support(self, batch_input, training=True):
        batch_idxs = batch_input["locs"][:, 0].int()
        p2v_map = batch_input["p2v_map"]
        locs_float = batch_input["locs_float"]
        batch_offsets = batch_input["batch_offsets"]
        support_mask = batch_input["support_masks"]

        batch_size = len(batch_offsets) - 1
        assert batch_size > 0

        with torch.no_grad():
            sparse_input = self.preprocess_input(batch_input, batch_size)

            """ Backbone net """
            output = self.input_conv(sparse_input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[p2v_map.long()]
            output_feats = output_feats.contiguous()

            mask_indices = torch.nonzero(support_mask == 1).view(-1)
            output_feats_ = output_feats[mask_indices]
            locs_float_ = locs_float[mask_indices]

            batch_idxs_ = batch_idxs[mask_indices]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)

            support_embeddings = []
            for b in range(batch_size):
                start = batch_offsets_[b]
                end = batch_offsets_[b + 1]

                locs_float_b = locs_float_[start:end, :].unsqueeze(0)
                output_feats_b = output_feats_[start:end, :].unsqueeze(0)
                # support_embedding = torch.mean(output_feats_b, dim=1) # channel

                context_locs_b, grouped_features, grouped_xyz, pre_enc_inds = self.set_aggregator.group_points(
                    locs_float_b.contiguous(), output_feats_b.transpose(1, 2).contiguous(), npoint_new=32
                )
                context_feats_b = self.set_aggregator.mlp(grouped_features, grouped_xyz, pooling="avg")
                context_feats_b = context_feats_b.transpose(1, 2)  # 1 x n_point x channel

                support_embedding = torch.mean(context_feats_b, dim=1)  # channel
                support_embeddings.append(support_embedding)
            support_embeddings = torch.cat(support_embeddings)  # batch x channel
            return support_embeddings

    def forward(self, support_dict, scene_dict, training=True, remember=False, support_embeddings=None):
        outputs = {}

        batch_idxs = scene_dict["locs"][:, 0].int()
        locs_float = scene_dict["locs_float"]
        batch_offsets = scene_dict["batch_offsets"]

        batch_size = len(batch_offsets) - 1
        assert batch_size > 0

        pc_dims = [
            scene_dict["pc_mins"],
            scene_dict["pc_maxs"],
        ]

        if remember:
            (
                context_locs,
                context_feats,
                pre_enc_inds,
                fg_idxs,
                batch_offsets,
                output_feats_,
                batch_idxs_,
                locs_float_,
                batch_offsets_,
                semantic_preds_,
                semantic_scores,
                query_locs,
                mask_features_,
                geo_dists,
            ) = self.cache_data

            outputs["semantic_scores"] = semantic_scores
        else:
            output_feats, semantic_scores, semantic_preds = self.forward_backbone(scene_dict, batch_size)

            outputs["semantic_scores"] = semantic_scores

            if cfg.train_fold == cfg.cvfold:
                fg_condition = semantic_preds >= 4
            else:
                fg_condition = semantic_preds == 3

            fg_idxs = torch.nonzero(fg_condition).view(-1)

            batch_idxs_ = batch_idxs[fg_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
            locs_float_ = locs_float[fg_idxs]
            output_feats_ = output_feats[fg_idxs]
            semantic_preds_ = semantic_preds[fg_idxs]

            context_mask_tower = (
                torch.enable_grad if self.training and "mask_tower" not in self.fix_module else torch.no_grad
            )
            with context_mask_tower():
                mask_features_ = self.mask_tower(torch.unsqueeze(output_feats_, dim=2).permute(2, 1, 0)).permute(
                    2, 1, 0
                )

            # NOTE aggregator
            contexts = self.forward_aggregator(locs_float_, output_feats_, batch_offsets_, batch_size)
            if contexts is None:
                outputs["mask_predictions"] = None
                return outputs

            context_locs, context_feats, pre_enc_inds = contexts

            # NOTE get queries
            query_locs = context_locs[:, : cfg.n_query_points, :]

            # NOTE process geodist
            geo_dists = cal_geodesic_vectorize(
                self.geo_knn,
                pre_enc_inds,
                locs_float_,
                batch_offsets_,
                max_step=128 if self.training else 256,
                neighbor=64,
                radius=0.05,
                n_queries=cfg.n_query_points,
            )

            self.cache_data = (
                context_locs,
                context_feats,
                pre_enc_inds,
                fg_idxs,
                batch_offsets,
                output_feats_,
                batch_idxs_,
                locs_float_,
                batch_offsets_,
                semantic_preds_,
                semantic_scores,
                query_locs,
                mask_features_,
                geo_dists,
            )

        if len(fg_idxs) == 0:
            outputs["proposal_scores"] = None
            return outputs

        if support_embeddings is None:
            support_embeddings = self.process_support(support_dict, training)  # batch x channel

        # NOTE aggregate support and query feats
        channel_wise_tensor = context_feats * support_embeddings.unsqueeze(1).repeat(1, cfg.n_decode_point, 1)
        subtraction_tensor = context_feats - support_embeddings.unsqueeze(1).repeat(1, cfg.n_decode_point, 1)
        aggregation_tensor = torch.cat(
            [channel_wise_tensor, subtraction_tensor, context_feats], dim=2
        )  # batch * n_sampling *(3*channel)

        # NOTE transformer decoder
        dec_outputs = self.forward_decoder(
            context_locs, aggregation_tensor, query_locs, pc_dims, geo_dists, pre_enc_inds
        )

        if not training:
            dec_outputs = dec_outputs[-1:, ...]

        # NOTE dynamic convolution
        mask_predictions = self.get_mask_prediction(
            geo_dists, dec_outputs, mask_features_, locs_float_, query_locs, batch_offsets_
        )

        mask_logit_final = mask_predictions[-1]["mask_logits"]

        # NOTE check similarity between support and anchor
        similarity_score, pre_enc_inds_mask = self.get_similarity(
            mask_logit_final,
            batch_offsets_,
            locs_float_,
            output_feats_,
            support_embeddings,
            pre_enc_inds_mask=self.cache_pre_enc_inds_mask if remember else None,
        )
        self.cache_pre_enc_inds_mask = pre_enc_inds_mask

        if training:
            outputs["fg_idxs"] = fg_idxs
            outputs["num_insts"] = cfg.n_query_points * batch_size
            outputs["batch_idxs"] = batch_idxs_
            outputs["simnet"] = similarity_score
            outputs["mask_predictions"] = mask_predictions

            return outputs

        similarity_score_sigmoid = similarity_score.detach().sigmoid()
        scores_final, proposals_pred = self.generate_proposal(
            mask_logit_final,
            similarity_score_sigmoid,
            fg_idxs,
            batch_offsets,
            logit_thresh=0.2,
            score_thresh=cfg.TEST_SCORE_THRESH,
            npoint_thresh=cfg.TEST_NPOINT_THRESH,
            sim_score_thresh=cfg.similarity_thresh,
        )
        outputs["proposal_scores"] = (scores_final, proposals_pred)
        return outputs

    def forward_backbone(self, batch_input, batch_size):
        context_backbone = torch.enable_grad if self.training and "unet" not in self.fix_module else torch.no_grad
        with context_backbone():
            p2v_map = batch_input["p2v_map"]

            sparse_input = self.preprocess_input(batch_input, batch_size)

            """ Backbone net """
            output = self.input_conv(sparse_input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[p2v_map.long()]
            output_feats = output_feats.contiguous()

            """ Semantic head"""
            semantic_feats = self.semantic(output_feats)
            semantic_scores = self.semantic_linear(semantic_feats)  # (N, nClass), float
            semantic_preds = semantic_scores.max(1)[1]  # (N), long

            return output_feats, semantic_scores, semantic_preds

    def forward_aggregator(self, locs_float_, output_feats_, batch_offsets_, batch_size):
        context_aggregator = (
            torch.enable_grad if self.training and "set_aggregator" not in self.fix_module else torch.no_grad
        )
        with context_aggregator():

            context_locs = []
            grouped_features = []
            grouped_xyz = []
            pre_enc_inds = []

            for b in range(batch_size):
                start = batch_offsets_[b]
                end = batch_offsets_[b + 1]
                locs_float_b = locs_float_[start:end, :]
                output_feats_b = output_feats_[start:end, :]
                batch_points = (end - start).item()

                if batch_points == 0:
                    return None

                locs_float_b = locs_float_b.unsqueeze(0)
                output_feats_b = output_feats_b.unsqueeze(0)

                context_locs_b, grouped_features_b, grouped_xyz_b, pre_enc_inds_b = self.set_aggregator.group_points(
                    locs_float_b.contiguous(), output_feats_b.transpose(1, 2).contiguous()
                )

                context_locs.append(context_locs_b)
                grouped_features.append(grouped_features_b)
                grouped_xyz.append(grouped_xyz_b)
                pre_enc_inds.append(pre_enc_inds_b)

            context_locs = torch.cat(context_locs)
            grouped_features = torch.cat(grouped_features)
            grouped_xyz = torch.cat(grouped_xyz)
            pre_enc_inds = torch.cat(pre_enc_inds)

            context_feats = self.set_aggregator.mlp(grouped_features, grouped_xyz)
            context_feats = context_feats.transpose(1, 2)

            return context_locs, context_feats, pre_enc_inds

    def forward_decoder(self, context_locs, context_feats, query_locs, pc_dims, geo_dists, pre_enc_inds):
        batch_size = context_locs.shape[0]

        context_embedding_pos = self.pos_embedding(context_locs, input_range=pc_dims)
        context_feats = self.encoder_to_decoder_projection(context_feats.permute(0, 2, 1))  # batch x channel x npoints

        """ Init dec_inputs by query features """
        query_embedding_pos = self.pos_embedding(query_locs, input_range=pc_dims)
        query_embedding_pos = self.query_projection(query_embedding_pos.float())

        dec_inputs = context_feats[:, :, : cfg.n_query_points].permute(2, 0, 1)

        # decoder expects: npoints x batch x channel
        context_embedding_pos = context_embedding_pos.permute(2, 0, 1)
        query_embedding_pos = query_embedding_pos.permute(2, 0, 1)
        context_feats = context_feats.permute(2, 0, 1)

        # Encode relative pos
        relative_coords = torch.abs(
            query_locs[:, :, None, :] - context_locs[:, None, :, :]
        )  # b x n_queries x n_contexts x 3
        n_queries, n_contexts = relative_coords.shape[1], relative_coords.shape[2]

        geo_dist_context = []
        for b in range(batch_size):
            geo_dist_context_b = geo_dists[b][:, pre_enc_inds[b].long()]  # n_queries x n_contexts
            geo_dist_context.append(geo_dist_context_b)

        geo_dist_context = torch.stack(geo_dist_context, dim=0)  # b x n_queries x n_contexts
        max_geo_dist_context = torch.max(geo_dist_context, dim=2)[0]  # b x n_queries
        max_geo_val = torch.max(max_geo_dist_context)
        max_geo_dist_context[max_geo_dist_context < 0] = max_geo_val  # NOTE assign very big value to invalid queries

        max_geo_dist_context = max_geo_dist_context[:, :, None, None].expand(
            batch_size, n_queries, n_contexts, 3
        )  # b x n_queries x n_contexts x 3

        geo_dist_context = geo_dist_context[:, :, :, None].repeat(1, 1, 1, 3)

        cond = geo_dist_context < 0
        geo_dist_context[cond] = max_geo_dist_context[cond] + relative_coords[cond]

        relative_embedding_pos = self.pos_embedding(
            geo_dist_context.reshape(batch_size, n_queries * n_contexts, -1), input_range=pc_dims
        ).reshape(
            batch_size,
            -1,
            n_queries,
            n_contexts,
        )
        relative_embedding_pos = relative_embedding_pos.permute(2, 3, 0, 1)

        # num_layers x n_queries x batch x channel
        dec_outputs = self.decoder(
            tgt=dec_inputs,
            memory=context_feats,
            pos=context_embedding_pos,
            query_pos=query_embedding_pos,
            relative_pos=relative_embedding_pos,
        )

        return dec_outputs

    def get_similarity(
        self, mask_logit_final, batch_offsets_, locs_float_, output_feats_, support_embeddings, pre_enc_inds_mask=None
    ):
        batch_size = len(mask_logit_final)

        no_cache = pre_enc_inds_mask is None
        with torch.no_grad():
            if no_cache:
                pre_enc_inds_mask = []
            final_mask_features_arr = []
            for b in range(batch_size):
                start = batch_offsets_[b]
                end = batch_offsets_[b + 1]

                locs_float_b = locs_float_[start:end, :].unsqueeze(0)
                output_feats_b = output_feats_[start:end, :].unsqueeze(0)  # 1, n_point, f

                npoint_new = min(4096, end - start)

                context_locs_b, grouped_features, grouped_xyz, pre_enc_inds_b1 = self.set_aggregator.group_points(
                    locs_float_b.contiguous(),
                    output_feats_b.transpose(1, 2).contiguous(),
                    npoint_new=npoint_new,
                    inds=None if no_cache else pre_enc_inds_mask[b],
                )

                context_feats_b1 = self.set_aggregator.mlp(grouped_features, grouped_xyz, pooling="avg")
                context_feats_b1 = context_feats_b1.transpose(1, 2)  # 1 x n_point x channel

                mask_logit_final_b = mask_logit_final[b].detach().sigmoid()  # n_queries, mask

                mask_logit_final_b = mask_logit_final_b[:, pre_enc_inds_b1.squeeze(0).long()]
                mask_logit_final_bool = mask_logit_final_b >= 0.2  # n_queries, mask

                count_mask = torch.sum(mask_logit_final_bool, dim=1).int()

                output_feats_b_expand = context_feats_b1.expand(
                    count_mask.shape[0], context_feats_b1.shape[1], context_feats_b1.shape[2]
                )  # n_queries, mask, f

                final_mask_features = torch.sum(
                    (output_feats_b_expand * mask_logit_final_bool[:, :, None]), dim=1
                )  # n_queries, f

                final_mask_features = final_mask_features / (count_mask[:, None] + 1e-6)  # n_queries, f

                final_mask_features[count_mask <= 1] = 0.0
                final_mask_features_arr.append(final_mask_features)
                if no_cache:
                    pre_enc_inds_mask.append(pre_enc_inds_b1)
            final_mask_features_arr = torch.stack(final_mask_features_arr, dim=0)  # batch, n_queries, f

        """ channel-wise correlate """
        channel_wise_tensor_sim = final_mask_features_arr * support_embeddings.unsqueeze(1).repeat(
            1, final_mask_features_arr.shape[1], 1
        )
        subtraction_tensor_sim = final_mask_features_arr - support_embeddings.unsqueeze(1).repeat(
            1, final_mask_features_arr.shape[1], 1
        )
        aggregation_tensor_sim = torch.cat(
            [channel_wise_tensor_sim, subtraction_tensor_sim, final_mask_features_arr], dim=2
        )
        similarity_score = (
            self.similarity_net(aggregation_tensor_sim.flatten(0, 1))
            .squeeze(-1)
            .reshape(batch_size, aggregation_tensor_sim.shape[1])
        )  # batch  x n_sampling

        return similarity_score, pre_enc_inds_mask
