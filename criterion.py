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


def compute_score_loss(inputs, targets, num_boxes):
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
    # prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return ce_loss.mean(1).sum() / (num_boxes + 1e-6)


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


class InstSetCriterion(nn.Module):
    def __init__(self):
        super(InstSetCriterion, self).__init__()

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)

        self.score_criterion = nn.BCELoss(reduction="none").cuda()

        self.batch_size = cfg.batch_size
        self.n_queries = cfg.n_query_points

        self.matcher = HungarianMatcher(self.batch_size, self.n_queries)

        self.loss_weight = {
            "dice_loss": 1,
            "focal_loss": 1,
            "cls_loss": 1,
        }

        self.cached = []

    def single_layer_loss(self, mask_prediction, instance_masked, semantic_masked, batch_ids, cal_match=False):
        loss = torch.tensor(0.0, requires_grad=True).to(instance_masked.device)
        loss_dict = {}

        mask_logits_list = mask_prediction["mask_logits"]  # list of n_queries x N_mask
        cls_logits = mask_prediction["cls_logits"]  # batch x n_queries x n_classes

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True).to(cls_logits.device)

        num_gt = 0
        for batch in range(self.batch_size):
            mask_logit_b = mask_logits_list[batch]
            cls_logit_b = cls_logits[batch]  # n_queries x n_classes
            instance_masked_b = instance_masked[batch_ids == batch]
            semantic_masked_b = semantic_masked[batch_ids == batch]

            if mask_logit_b is None:
                continue

            if cal_match:
                mask_logit_b_detach = mask_logit_b.detach()
                cls_logit_b_detach = cls_logit_b.detach()
                pred_inds, inst_mask_gt, sem_cls_gt = self.matcher.forward_seg_single(
                    mask_logit_b_detach, cls_logit_b_detach, instance_masked_b, semantic_masked_b
                )

                self.cached.append((pred_inds, inst_mask_gt, sem_cls_gt))
            else:
                pred_inds, inst_mask_gt, sem_cls_gt = self.cached[batch]

            if pred_inds is None:
                continue
            mask_logit_pred = mask_logit_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch

            loss_dict["dice_loss"] += compute_dice_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            loss_dict["focal_loss"] += compute_sigmoid_focal_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            cls_label = torch.zeros((self.n_queries)).to(cls_logits.device)
            cls_label[pred_inds] = sem_cls_gt

            loss_dict["cls_loss"] += F.cross_entropy(
                cls_logit_b,
                cls_label.long(),
                reduction="mean",
            )

        for k in self.loss_weight:
            loss_dict[k] = loss_dict[k] * self.loss_weight[k] / self.batch_size
            loss += loss_dict[k]

        return loss, loss_dict, num_gt

    def forward(self, model_outputs, batch_inputs, epoch):

        # '''semantic loss'''
        # semantic_scores = model_outputs['semantic_scores']
        semantic_scores = model_outputs["semantic_scores"]
        semantic_labels = batch_inputs["labels"]
        instance_labels = batch_inputs["instance_labels"]

        loss_dict_out = {}
        loss = torch.tensor(0.0, requires_grad=True).to(semantic_scores.device)

        if "semantic" not in cfg.fix_module:
            semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        else:
            semantic_loss = torch.tensor(0.0, requires_grad=True).to(semantic_scores.device)

        loss += semantic_loss

        if epoch <= cfg.prepare_epochs:
            loss_dict_out["sem_loss"] = (semantic_loss.item(), semantic_labels.shape[0])
            loss_dict_out["loss"] = (loss.item(), semantic_labels.shape[0])
            return loss, loss_dict_out

        mask_predictions = model_outputs["mask_predictions"]
        fg_idxs = model_outputs["fg_idxs"]
        # num_insts   = model_outputs['num_insts']

        instance_masked = instance_labels[fg_idxs]
        semantic_masked = semantic_labels[fg_idxs]

        batch_ids = model_outputs["batch_idxs"]

        """ Main loss """
        self.cached = []
        main_loss, loss_dict, num_gt = self.single_layer_loss(
            mask_predictions[-1], instance_masked, semantic_masked, batch_ids, cal_match=True
        )

        loss += main_loss

        """ Auxilary loss """
        for l in range(cfg.dec_nlayers - 1):
            interm_loss, _, _ = self.single_layer_loss(
                mask_predictions[l], instance_masked, semantic_masked, batch_ids
            )
            loss += interm_loss

        loss_dict_out["focal_loss"] = (loss_dict["focal_loss"].item(), num_gt)
        loss_dict_out["dice_loss"] = (loss_dict["dice_loss"].item(), num_gt)
        loss_dict_out["cls_loss"] = (loss_dict["cls_loss"].item(), self.n_queries)
        loss_dict_out["sem_loss"] = (semantic_loss.item(), semantic_labels.shape[0])
        loss_dict_out["loss"] = (loss.item(), semantic_labels.shape[0])

        return loss, loss_dict_out
