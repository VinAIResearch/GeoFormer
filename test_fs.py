import os
import random

import numpy as np
import torch
from util.config import cfg
import time

import util.eval as eval
from checkpoint import align_and_update_state_dicts, strip_prefix_if_present
from datasets.scannetv2 import BENCHMARK_SEMANTIC_LABELS

from model.geoformer.geoformer_fs import GeoFormerFS
from datasets.scannetv2_fs_inst import FSInstDataset
from lib.pointgroup_ops.functions import pointgroup_ops
from util.log import create_logger
from util.utils_3d import load_ids, non_max_suppression_gpu


def init():
    os.makedirs(cfg.exp_path, exist_ok=True)

    global logger
    logger = create_logger(task="test")
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def load_set_support(model, dataset):
    set_support_name = cfg.type_support + str(cfg.cvfold) + "_" + str(cfg.k_shot) + "shot_10sets.pth"
    set_support_file = os.path.join("exp", cfg.file_support, set_support_name)

    # print(set_support_file)
    # if os.path.exists(set_support_file):
    #     logger.info("Found set_support_vector.")
    #     set_support_vectors = torch.load(set_support_file)
    #     return set_support_vectors

    # os.makedirs(os.path.join('exp', cfg.file_support), exist_ok=True)
    logger.info(f"Generate support vectors and save to {set_support_file}")
    dataset.genSupportLoader()
    model.eval()
    net_device = next(model.parameters()).device
    set_support_vectors = []
    with torch.no_grad():
        for subset in range(cfg.run_num):

            support_vector = {}
            support_set = dataset.support_set[subset]
            for cls in dataset.SEMANTIC_LABELS:
                sup_vectors = []
                list_scenes = support_set[cls]
                for i in range(cfg.k_shot):
                    support_tuple = list_scenes[i]

                    support_scene_name, support_instance_id = support_tuple[0], support_tuple[1]
                    (
                        support_xyz_middle,
                        support_xyz_scaled,
                        support_rgb,
                        support_label,
                        support_instance_label,
                    ) = dataset.load_single(support_scene_name, aug=False, permutate=False, val=True, support=True)

                    support_mask = (support_instance_label == support_instance_id).astype(int)

                    support_batch_offsets = torch.tensor([0, support_xyz_middle.shape[0]], dtype=torch.int)
                    support_masks_offset = torch.tensor(
                        [0, np.count_nonzero(support_mask)], dtype=torch.int
                    )  # int (B+1)
                    support_locs = torch.cat(
                        [
                            torch.LongTensor(support_xyz_scaled.shape[0], 1).fill_(0),
                            torch.from_numpy(support_xyz_scaled).long(),
                        ],
                        1,
                    )
                    support_locs_float = torch.from_numpy(support_xyz_middle).to(torch.float32)
                    support_feats = torch.from_numpy(support_rgb).to(torch.float32)  # float (N, C)
                    support_masks = torch.from_numpy(support_mask)
                    support_spatial_shape = np.clip((support_locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

                    # voxelize
                    support_voxel_locs, support_p2v_map, support_v2p_map = pointgroup_ops.voxelization_idx(
                        support_locs, 1, dataset.mode
                    )

                    support_dict = {
                        "voxel_locs": support_voxel_locs,
                        "p2v_map": support_p2v_map,
                        "v2p_map": support_v2p_map,
                        "locs": support_locs,
                        "locs_float": support_locs_float,
                        "feats": support_feats,
                        "support_masks": support_masks,
                        "spatial_shape": support_spatial_shape,
                        "batch_offsets": support_batch_offsets,
                        "mask_offsets": support_masks_offset,
                    }

                    for key in support_dict:
                        if torch.is_tensor(support_dict[key]):
                            support_dict[key] = support_dict[key].to(net_device)

                    sup_vec = model.process_support(support_dict, training=False)
                    sup_vectors.append(sup_vec)
                sup_vectors = torch.cat(sup_vectors, dim=0)
                mean_vector = torch.mean(sup_vectors, dim=0)
                support_vector[cls] = mean_vector.cpu()
            set_support_vectors.append(support_vector)

    # torch.save(set_support_vectors, set_support_file)
    logger.info("Finish create support vectors")
    return set_support_vectors


def do_test(model, dataset):
    model.eval()
    net_device = next(model.parameters()).device

    set_support_vectors = load_set_support(model, dataset)

    logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")
    dataloader = dataset.testLoader()

    num_test_scenes = len(dataloader)

    with torch.no_grad():

        gt_file_arr = []
        test_scene_name_arr = []
        pred_info_arr = [[] for idx in range(cfg.run_num)]

        start_time = time.time()
        for i, batch_input in enumerate(dataloader):
            nclusters = [0] * cfg.run_num
            clusters = [[] for idx in range(cfg.run_num)]
            cluster_scores = [[] for idx in range(cfg.run_num)]
            cluster_semantic_id = [[] for idx in range(cfg.run_num)]
            is_valid, list_support_dicts, query_dict, scene_infos = batch_input
            if not is_valid:
                continue

            test_scene_name = scene_infos["query_scene"]
            active_label = scene_infos["active_label"]

            N = query_dict["feats"].shape[0]

            for key in query_dict:
                if torch.is_tensor(query_dict[key]):
                    query_dict[key] = query_dict[key].to(net_device)

            for j, (label, support_dict) in enumerate(zip(active_label, list_support_dicts)):
                for k in range(cfg.run_num):  # NOTE number of runs
                    remember = False if (j == 0 and k == 0) else True

                    support_embeddings = None
                    if cfg.fix_support:
                        support_embeddings = set_support_vectors[k][label].unsqueeze(0).to(net_device)
                    else:
                        for key in support_dict:
                            if torch.is_tensor(support_dict[key]):
                                support_dict[key] = support_dict[key].to(net_device)

                    outputs = model(
                        support_dict,
                        query_dict,
                        training=False,
                        remember=remember,
                        support_embeddings=support_embeddings,
                    )

                    if outputs["proposal_scores"] is None:
                        continue
                    scores_pred, proposals_pred = outputs["proposal_scores"]
                    if isinstance(scores_pred, list):
                        continue

                    benchmark_label = BENCHMARK_SEMANTIC_LABELS[label]
                    cluster_semantic = torch.ones((proposals_pred.shape[0], 1)) * benchmark_label

                    clusters[k].append(proposals_pred)
                    cluster_scores[k].append(scores_pred)
                    cluster_semantic_id[k].append(cluster_semantic)

                    # torch.cuda.empty_cache()

            test_scene_name_arr.append(test_scene_name)
            gt_file_name = os.path.join(cfg.data_root, cfg.dataset, "val_gt", test_scene_name + ".txt")
            gt_file_arr.append(gt_file_name)

            for k in range(cfg.run_num):
                if len(clusters[k]) == 0:
                    pred_info_arr[k].append(None)
                    continue
                clusters[k] = torch.cat(clusters[k], axis=0)
                cluster_scores[k] = torch.cat(cluster_scores[k], axis=0)
                cluster_semantic_id[k] = torch.cat(cluster_semantic_id[k], axis=0)

                # nms
                if cluster_scores[k].shape[0] == 0:
                    pick_idxs_cluster = np.empty(0)
                else:
                    clusters_f = clusters[k].float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(clusters_f, clusters_f.t())  # (nProposal, nProposal), float, cuda
                    clusters_pointnum = clusters_f.sum(1)  # (nProposal), float, cuda
                    clusters_pn_h = clusters_pointnum.unsqueeze(-1).repeat(1, clusters_pointnum.shape[0])
                    clusters_pn_v = clusters_pointnum.unsqueeze(0).repeat(clusters_pointnum.shape[0], 1)
                    cross_ious = intersection / (clusters_pn_h + clusters_pn_v - intersection)
                    pick_idxs_cluster = non_max_suppression_gpu(
                        cross_ious, cluster_scores[k], cfg.TEST_NMS_THRESH
                    )  # int, (nCluster, N)

                clusters[k] = clusters[k][pick_idxs_cluster].cpu().numpy()
                cluster_scores[k] = cluster_scores[k][pick_idxs_cluster].cpu().numpy()
                cluster_semantic_id[k] = cluster_semantic_id[k][pick_idxs_cluster].cpu().numpy()
                nclusters[k] = clusters[k].shape[0]

                if cfg.eval:
                    pred_info = {}
                    pred_info["conf"] = cluster_scores[k]
                    pred_info["label_id"] = cluster_semantic_id[k]
                    pred_info["mask"] = clusters[k]
                    pred_info_arr[k].append(pred_info)

            overlap_time = time.time() - start_time

            logger.info(
                f"Test scene {i+1}/{num_test_scenes}: {test_scene_name} | Elapsed time: {int(overlap_time)}s | Remaining time: {int(overlap_time * float(num_test_scenes-(i+1))/(i+1))}s"
            )
            logger.info(f"Num points: {N} | Num instances of {cfg.run_num} runs: {nclusters}")

        # evaluation
        if cfg.eval:
            logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
            run_dict = {}
            for k in range(cfg.run_num):
                matches = {}
                for i in range(len(pred_info_arr[k])):
                    pred_info = pred_info_arr[k][i]
                    if pred_info is None:
                        continue

                    gt_file_name = gt_file_arr[i]
                    test_scene_name = test_scene_name_arr[i]
                    gt_ids = load_ids(gt_file_name)

                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_ids)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]["gt"] = gt2pred
                    matches[test_scene_name]["pred"] = pred2gt

                ap_scores = eval.evaluate_matches(matches)
                avgs = eval.compute_averages(ap_scores)
                eval.print_results(avgs, logger)
                run_dict = eval.accumulate_averages_over_runs(run_dict, avgs)

            run_dict = eval.compute_averages_over_runs(run_dict)
            eval.print_results(run_dict, logger)


if __name__ == "__main__":
    init()

    # model
    logger.info("=> creating model ...")

    model = GeoFormerFS()
    model = model.cuda()

    logger.info("# parameters (model): {}".format(sum([x.nelement() for x in model.parameters()])))

    checkpoint_fn = cfg.resume
    if os.path.isfile(checkpoint_fn):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
        state = torch.load(checkpoint_fn)
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)

        logger.info("=> loaded checkpoint '{}')".format(checkpoint_fn))
    else:
        raise RuntimeError

    dataset = FSInstDataset(split_set="val")

    # evaluate
    do_test(
        model,
        dataset,
    )
