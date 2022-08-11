import datetime
import os
import time

import numpy as np
import torch
import torch.optim as optim
import util.utils as utils
from checkpoint import align_and_update_state_dicts, checkpoint, strip_prefix_if_present
from criterion_fs import FSInstSetCriterion
from datasets.scannetv2_fs_inst import FSInstDataset
from model.geoformer.geoformer_fs import GeoFormerFS

from tensorboardX import SummaryWriter
from util.config import cfg
from util.dataloader_util import get_rank
from util.log import create_logger
from util.utils_scheduler import adjust_learning_rate


def init():
    os.makedirs(cfg.exp_path, exist_ok=True)
    # log the config
    global logger
    logger = create_logger()
    logger.info(cfg)
    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)


def train_one_epoch(epoch, train_loader, model, criterion, optimizer):

    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    # model.set_eval()
    net_device = next(model.parameters()).device

    num_iter = len(train_loader)
    max_iter = cfg.epochs * num_iter

    start_time = time.time()
    check_time = time.time()

    for iteration, batch in enumerate(train_loader):
        data_time.update(time.time() - check_time)
        torch.cuda.empty_cache()
        current_iter = (epoch - 1) * num_iter + iteration + 1
        remain_iter = max_iter - current_iter

        curr_lr = adjust_learning_rate(optimizer, current_iter / max_iter, cfg.epochs)

        support_dict, query_dict, scene_infos = batch

        for key in support_dict:
            if torch.is_tensor(support_dict[key]):
                support_dict[key] = support_dict[key].to(net_device)
        for key in query_dict:
            if torch.is_tensor(query_dict[key]):
                query_dict[key] = query_dict[key].to(net_device)

        outputs = model(support_dict, query_dict, remember=False, training=True)

        if "mask_predictions" not in outputs.keys() or outputs["mask_predictions"] is None:
            continue

        loss, loss_dict = criterion(outputs, query_dict, epoch)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_time.update(time.time() - check_time)
        check_time = time.time()

        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        for k, v in loss_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])
            writer.add_scalar("Loss/" + k, v[0], iteration)

        del loss, outputs, loss_dict
        if iteration % 10 == 0:
            if epoch <= cfg.prepare_epochs:
                logger.info(
                    "Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Focal loss: {:.4f}({:.4f}) | Dice loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format(
                        epoch,
                        cfg.epochs,
                        iteration + 1,
                        num_iter,
                        curr_lr,
                        am_dict["loss"].val,
                        am_dict["loss"].avg,
                        am_dict["focal_loss"].val,
                        am_dict["focal_loss"].avg,
                        am_dict["dice_loss"].val,
                        am_dict["dice_loss"].avg,
                        mem_mb,
                        iter_time.val,
                        remain_time=remain_time,
                    )
                )
            else:
                # logger.info("Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Sim loss: {:.4f}({:.4f}) | Focal loss: {:.4f}({:.4f}) | Dice loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format
                #     (epoch, cfg.epochs, iteration + 1, num_iter, curr_lr,
                #     am_dict['loss'].val, am_dict['loss'].avg,
                #     am_dict['sim_loss'].val, am_dict['sim_loss'].avg,
                #     am_dict['focal_loss'].val, am_dict['focal_loss'].avg,
                #     am_dict['dice_loss'].val, am_dict['dice_loss'].avg,
                #     mem_mb,
                #     iter_time.val, remain_time=remain_time))

                logger.info(
                    "Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Focal loss: {:.4f}({:.4f}) | Dice loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format(
                        epoch,
                        cfg.epochs,
                        iteration + 1,
                        num_iter,
                        curr_lr,
                        am_dict["loss"].val,
                        am_dict["loss"].avg,
                        # am_dict['sim_loss'].val, am_dict['sim_loss'].avg,
                        am_dict["focal_loss"].val,
                        am_dict["focal_loss"].avg,
                        am_dict["dice_loss"].val,
                        am_dict["dice_loss"].avg,
                        mem_mb,
                        iter_time.val,
                        remain_time=remain_time,
                    )
                )

    if epoch % cfg.save_freq == 0 or iteration == cfg.epochs:
        checkpoint(model, optimizer, epoch, cfg.output_path, None, None)
    checkpoint(model, optimizer, epoch, cfg.output_path, None, None, last=True)

    for k in am_dict.keys():
        writer.add_scalar(k + "_train", am_dict[k].avg, epoch)

    logger.info(
        "epoch: {}/{}, train loss: {:.4f}, time: {}s".format(
            epoch, cfg.epochs, am_dict["loss"].avg, time.time() - start_time
        )
    )
    logger.info("=========================================")


if __name__ == "__main__":
    # init
    init()

    torch.cuda.set_device(0)
    np.random.seed(cfg.manual_seed + get_rank())
    torch.manual_seed(cfg.manual_seed + get_rank())
    torch.cuda.manual_seed_all(cfg.manual_seed + get_rank())

    # model
    logger.info("=> creating model ...")

    model = GeoFormerFS()
    model = model.cuda()

    logger.info("# training parameters: {}".format(sum([x.nelement() for x in model.parameters() if x.requires_grad])))

    criterion = FSInstSetCriterion()
    criterion = criterion.cuda()

    # optimizer
    if cfg.optim == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    logger.info(f"Learning rate: {cfg.lr}")
    start_epoch = -1

    if cfg.pretrain:
        logger.info("=> loading checkpoint '{}'".format(cfg.pretrain))
        loaded = torch.load(cfg.pretrain, map_location=torch.device("cpu"))["state_dict"]
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(loaded, prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)
        logger.info("=> done loading pretrain")

    if cfg.resume:
        checkpoint_fn = cfg.resume
        if os.path.isfile(checkpoint_fn):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
            start_epoch = state["epoch"] + 1

            model_state_dict = model.state_dict()
            loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model.load_state_dict(model_state_dict)

            logger.info("=> loaded checkpoint '{}' (start_epoch {})".format(checkpoint_fn, start_epoch))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    dataset = FSInstDataset(split_set="train")
    train_loader = dataset.trainLoader()

    if start_epoch == -1:
        start_epoch = 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_one_epoch(epoch, train_loader, model, criterion, optimizer)
