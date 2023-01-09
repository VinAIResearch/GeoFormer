# ScanNet util_3d: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util_3d.py

import json

import numpy as np
import torch


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


# ------------ Instance Utils ------------ #


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id = int(data["instance_id"])
        self.label_id = int(data["label_id"])
        self.vert_count = int(data["vert_count"])
        if "med_dist" in data:
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return "(" + str(self.instance_id) + ")"


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances


def non_max_suppression_gpu(ious, scores, threshold):
    ixs = torch.argsort(scores, descending=True)

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]

        remove_ixs = torch.nonzero(iou > threshold).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]
        
    return torch.tensor(pick, dtype=torch.long, device=scores.device)

def matrix_non_max_suppression(proposals_pred, scores, categories, kernel='gaussian', sigma=2.0, final_score_thresh=0.05):
    ixs = torch.argsort(scores, descending=True)
    # ixs = ixs[:max_prev_nms]
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    proposals_pred_sorted = proposals_pred[ixs]
    scores_sorted = scores[ixs]

    # (nProposal, N), float, cuda
    intersection = torch.einsum("nc,mc->nm", proposals_pred_sorted.type(scores.dtype), proposals_pred_sorted.type(scores.dtype))
    proposals_pointnum = proposals_pred_sorted.sum(1)  # (nProposal), float, cuda
    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
    ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)

    

    # label_specific matrix.
    categories_x = categories_sorted[None,:].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = ious * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient

    # print('cate_scores_update', cate_scores_update)
    score_mask = (cate_scores_update >= final_score_thresh)

    return ixs[score_mask]

