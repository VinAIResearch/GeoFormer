# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_box_cdist
import functools

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


print = functools.partial(print, flush=True)


def compute_dice(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, batch_size, n_queries, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.batch_size = batch_size
        self.n_queries = n_queries

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward_seg_single(self, mask_logit, sem_logit, instance_masked, semantic_masked, fewshot=False):
        with torch.no_grad():

            n_mask = instance_masked.shape[-1]

            if n_mask == 0:
                return None, None, None

            unique_inst = sorted(torch.unique(instance_masked))
            unique_inst = [i for i in unique_inst if i != -100]
            n_inst_gt = len(unique_inst)
            # print('unique_inst', unique_inst)
            inst_masks = torch.zeros((n_inst_gt, n_mask)).to(mask_logit.device)
            sem_labels = torch.zeros((n_inst_gt)).to(mask_logit.device)
            # min_inst_id = min(unique_inst)
            count = 0
            for idx in unique_inst:
                temp = instance_masked == idx
                inst_masks[count, :] = temp

                sem_labels[count] = semantic_masked[torch.nonzero(temp)[0]]
                count += 1

            dice_cost = compute_dice(
                mask_logit.reshape(-1, 1, n_mask).repeat(1, n_inst_gt, 1).flatten(0, 1),
                inst_masks.reshape(1, -1, n_mask).repeat(self.n_queries, 1, 1).flatten(0, 1),
            )

            dice_cost = dice_cost.reshape(self.n_queries, n_inst_gt)

            # if torch.any(torch.isnan(dice_cost)):
            #     breakpoint()

            if fewshot:
                final_cost = 1 * dice_cost
            else:
                sem_logit = torch.nn.functional.softmax(sem_logit, dim=-1)
                class_cost = -torch.gather(
                    sem_logit, 1, sem_labels.unsqueeze(0).expand(self.n_queries, n_inst_gt).long()
                )

                final_cost = 1 * class_cost + 1 * dice_cost

            final_cost = final_cost.detach().cpu().numpy()

            row_inds, col_inds = linear_sum_assignment(final_cost)

            return row_inds, inst_masks[col_inds], sem_labels[col_inds]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
