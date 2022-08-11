import math
from math import cos, pi

from util.config import cfg


def compute_learning_rate(curr_epoch_normalized, max_epochs):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if curr_epoch_normalized <= (cfg.warm_lr_epochs / max_epochs) and cfg.warm_lr_epochs > 0:
        # Linear Warmup
        curr_lr = cfg.warm_lr + curr_epoch_normalized * max_epochs * ((cfg.base_lr - cfg.warm_lr) / cfg.warm_lr_epochs)
    else:
        # Cosine Learning Rate Schedule
        curr_lr = cfg.final_lr + 0.5 * (cfg.base_lr - cfg.final_lr) * (1 + math.cos(math.pi * curr_epoch_normalized))
    return curr_lr


def adjust_learning_rate(optimizer, curr_epoch, max_epochs):
    curr_lr = compute_learning_rate(curr_epoch, max_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
