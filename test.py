import os
import random
import time

import numpy as np
import torch
from util.config import cfg

import util.eval as eval
from checkpoint import align_and_update_state_dicts, strip_prefix_if_present
from datasets.scannetv2 import BENCHMARK_SEMANTIC_LABELS, FOLD
from datasets.scannetv2_inst import InstDataset
from model.geoformer.geoformer import GeoFormer
from util.log import create_logger
from util.utils_3d import load_ids, non_max_suppression_gpu


def init():
    global result_dir
    result_dir = cfg.exp_path
    os.makedirs(cfg.exp_path, exist_ok=True)

    global logger
    logger = create_logger(task="test")
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def do_test(model, dataloader, cur_epoch):

    model.eval()
    net_device = next(model.parameters()).device

    logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")
    num_test_scenes = len(dataloader)

    with torch.no_grad():
        gt_file_arr = []
        test_scene_name_arr = []
        pred_info_arr = []

        start_time = time.time()
        for i, batch_input in enumerate(dataloader):
            N = batch_input["feats"].shape[0]
            test_scene_name = batch_input["test_scene_name"][0]
            torch.cuda.empty_cache()

            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to(net_device)

            outputs = model(batch_input, cur_epoch, training=False)

            if "proposal_scores" not in outputs.keys():
                continue

            cls_final, scores_final, masks_final = outputs["proposal_scores"]  # (nProposal, 1) float, cuda
            if isinstance(cls_final, list):
                continue

            temp = torch.tensor(FOLD[cfg.cvfold], device=scores_final.device)[cls_final - 4]
            semantic_id = torch.tensor(BENCHMARK_SEMANTIC_LABELS, device=scores_final.device)[
                temp
            ]  # (nProposal), long

            test_scene_name_arr.append(test_scene_name)
            gt_file_name = os.path.join(cfg.data_root, cfg.dataset, "val_gt", test_scene_name + ".txt")
            gt_file_arr.append(gt_file_name)

            # nms
            if semantic_id.shape[0] == 0:
                pick_idxs = np.empty(0)
            else:
                proposals_pred_f = masks_final.float()  # (nProposal, N), float, cuda
                intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                pick_idxs = non_max_suppression_gpu(
                    cross_ious, scores_final, cfg.TEST_NMS_THRESH
                )  # int, (nCluster, N)

            clusters = masks_final[pick_idxs].cpu().numpy()
            cluster_scores = scores_final[pick_idxs].cpu().numpy()
            cluster_semantic_id = semantic_id[pick_idxs].cpu().numpy()
            nclusters = clusters.shape[0]

            if cfg.eval:
                pred_info = {}
                pred_info["conf"] = cluster_scores
                pred_info["label_id"] = cluster_semantic_id
                pred_info["mask"] = clusters
                pred_info_arr.append(pred_info)

            overlap_time = time.time() - start_time
            logger.info(
                f"Test scene {i+1}/{num_test_scenes}: {test_scene_name} | Elapsed time: {int(overlap_time)}s | Remaining time: {int(overlap_time * float(num_test_scenes-(i+1))/(i+1))}s"
            )
            logger.info(f"Num points: {N} | Num instances: {nclusters}")

        # evaluation
        if cfg.eval:
            logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

            matches = {}
            for i in range(len(pred_info_arr)):
                pred_info = pred_info_arr[i]
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


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == "__main__":
    init()

    # model
    logger.info("=> creating model ...")
    model = GeoFormer()
    model = model.cuda(0)

    # logger.info(model)
    logger.info("# parameters (model): {}".format(sum([x.nelement() for x in model.parameters()])))

    checkpoint_fn = cfg.resume
    if os.path.isfile(checkpoint_fn):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
        state = torch.load(checkpoint_fn)

        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)

    else:
        raise RuntimeError

    dataset = InstDataset(split_set="val")
    test_loader = dataset.testLoader()

    cur_epoch = 300
    # evaluate
    do_test(model, test_loader, cur_epoch)
