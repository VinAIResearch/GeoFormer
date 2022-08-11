import torch
import torch.nn as nn
import torch.nn.functional as F
from model.matcher import HungarianMatcher
from util.config import cfg


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = 1
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x**2.0).sum(dim=1) + (target**2.0).sum(dim=1) + eps
    loss = 1.0 - (2 * intersection / union)
    return loss


def compute_dice_loss(inputs, targets, num_boxes):
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
    return loss.sum() / (num_boxes + 1e-6)


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


class FocalLossV1(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        reduction="mean",
    ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, label):
        pred = pred.sigmoid()
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        alpha_factor = torch.ones(pred.shape).cuda() * self.alpha
        alpha_factor = torch.where(torch.eq(label, 1.0), alpha_factor, 1.0 - alpha_factor)
        focal_weight = torch.where(torch.eq(label, 1.0), 1.0 - pred, pred)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = -(label * torch.log(pred) + (1.0 - label) * torch.log(1.0 - pred))

        cls_loss = focal_weight * bce
        cls_loss = cls_loss.sum() / (label.shape[0] + 1e-6)

        return cls_loss


class FSInstSetCriterion(nn.Module):
    def __init__(self, cal_simloss=True):
        super(FSInstSetCriterion, self).__init__()

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
        self.score_criterion = nn.BCELoss(reduction="none")
        self.similarity_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.batch_size = cfg.batch_size
        self.n_queries = cfg.n_query_points

        self.matcher = HungarianMatcher(self.batch_size, self.n_queries)

        self.loss_weight = {
            "dice_loss": 1,
            "focal_loss": 1,
            # 'cls_loss': 1,
        }

        self.cal_simloss = "similarity_net" not in cfg.fix_module

        self.cached = []

    def sim_loss(self, similarity_score, instance_masked, mask_logits, batch_ids):
        train_label = torch.zeros((self.batch_size, self.n_queries)).cuda()
        n_hard_negatives = torch.zeros(self.batch_size).cuda()
        for b in range(self.batch_size):
            num_positive = 0
            num_negative = 0
            positive_inds = []
            negative_inds = []
            scene_instance_labels_b_ = instance_masked[batch_ids == b]  # n_mask
            mask_logits_b = mask_logits[b].clone().detach()
            mask_logits_b = (mask_logits_b.sigmoid() > 0.5).long()
            for n in range(cfg.n_query_points):
                mask_logits_b_n = mask_logits_b[n].long()  # n_mask
                mask_points = torch.nonzero(mask_logits_b_n).squeeze(-1)
                if len(mask_points) == 0:
                    num_negative += 1
                    negative_inds.append(n)
                    continue

                inst_label = torch.mode(scene_instance_labels_b_[mask_points])[0].item()
                if inst_label == -100:
                    num_negative += 1
                    negative_inds.append(n)
                    continue

                mask_logits_label = (scene_instance_labels_b_ == inst_label).long()
                intersection = ((mask_logits_b_n + mask_logits_label) > 1).long().sum()
                union = ((mask_logits_b_n + mask_logits_label) > 0).long().sum()
                iou = torch.true_divide(intersection, union)

                if iou >= 0.5:
                    num_positive += 1
                    positive_inds.append(n)
                elif iou <= 0.3:
                    num_negative += 1
                    negative_inds.append(n)

            if num_negative > cfg.negative_ratio * num_positive:
                n_hard_negatives[b] = cfg.negative_ratio * num_positive
            else:
                n_hard_negatives[b] = num_negative

            train_label[b, positive_inds] = 1
            # print('train_label', train_label)
        if train_label.sum() == 0:
            return torch.tensor(0.0, requires_grad=True).to(similarity_score.device)

        loss_all = self.similarity_criterion(similarity_score, train_label)
        loss_neg = loss_all.clone()

        loss_pos = loss_all * train_label

        loss_neg[train_label.long()] = 0
        loss_neg, _ = loss_neg.sort(dim=1, descending=True)

        hardness_ranks = (
            torch.LongTensor(range(cfg.n_query_points)).unsqueeze(0).expand_as(loss_neg).cuda()
        )  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        loss_hard_neg = loss_neg[hard_negatives]
        similarity_loss = (loss_hard_neg.sum() + loss_pos.sum()) / train_label.sum().float()

        return similarity_loss

    def single_layer_loss(
        self, mask_prediction, similarity_score, instance_masked, semantic_masked, batch_ids, cal_match=False
    ):
        loss = torch.tensor(0.0, requires_grad=True).to(instance_masked.device)
        loss_dict = {}

        mask_logits_list = mask_prediction["mask_logits"]  # list of n_queries x N_mask

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True).to(similarity_score.device)

        num_gt = 0
        for batch in range(self.batch_size):
            mask_logit_b = mask_logits_list[batch]
            similarity_score_b = similarity_score[batch]  # n_queries x n_classes
            instance_masked_b = instance_masked[batch_ids == batch]
            semantic_masked_b = semantic_masked[batch_ids == batch]

            if mask_logit_b is None:
                continue

            if cal_match:
                pred_inds, inst_mask_gt, sem_cls_gt = self.matcher.forward_seg_single(
                    mask_logit_b, similarity_score_b, instance_masked_b, semantic_masked_b, fewshot=True
                )

                self.cached.append((pred_inds, inst_mask_gt, sem_cls_gt))
            else:
                pred_inds, inst_mask_gt, sem_cls_gt = self.cached[batch]

            if pred_inds is None:
                continue
            mask_logit_pred = mask_logit_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch

            if num_gt_batch == 0:
                continue

            loss_dict["dice_loss"] += compute_dice_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            loss_dict["focal_loss"] += compute_sigmoid_focal_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)

        for k in self.loss_weight:
            loss_dict[k] = loss_dict[k] * self.loss_weight[k] / self.batch_size
            loss += loss_dict[k]

        return loss, loss_dict, num_gt

    def forward(self, model_outputs, batch_inputs, epoch):
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        loss_dict_out = {}

        mask_predictions = model_outputs["mask_predictions"]
        fg_idxs = model_outputs["fg_idxs"]

        instance_labels = batch_inputs["instance_labels"]
        semantic_labels = batch_inputs["labels"]

        instance_masked = instance_labels[fg_idxs]
        semantic_masked = semantic_labels[fg_idxs]

        batch_ids = model_outputs["batch_idxs"]

        mask_logits = mask_predictions[-1]["mask_logits"]

        similarity_score = model_outputs["simnet"]

        if epoch > cfg.prepare_epochs and self.cal_simloss:
            sim_loss = self.sim_loss(similarity_score, instance_masked, mask_logits, batch_ids)
            loss += sim_loss
            loss_dict_out["sim_loss"] = (sim_loss.item(), self.n_queries)

        """ Main loss """
        self.cached = []
        main_loss, loss_dict, num_gt = self.single_layer_loss(
            mask_predictions[-1], similarity_score, instance_masked, semantic_masked, batch_ids, cal_match=True
        )
        loss += main_loss

        """ Auxilary loss """
        for l in range(cfg.dec_nlayers - 1):
            interm_loss, _, _ = self.single_layer_loss(
                mask_predictions[l], similarity_score, instance_masked, semantic_masked, batch_ids
            )
            loss += interm_loss

        loss_dict_out["focal_loss"] = (loss_dict["focal_loss"].item(), num_gt)
        loss_dict_out["dice_loss"] = (loss_dict["dice_loss"].item(), num_gt)
        loss_dict_out["loss"] = (loss.item(), semantic_labels.shape[0])
        return loss, loss_dict_out
