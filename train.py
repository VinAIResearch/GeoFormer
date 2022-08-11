import datetime
import os
import time

import numpy as np
import torch
import torch.optim as optim
import util.utils as utils
from checkpoint import align_and_update_state_dicts, checkpoint, strip_prefix_if_present
from criterion import InstSetCriterion
from datasets.scannetv2_inst import InstDataset
from model.geoformer.geoformer import GeoFormer
from tensorboardX import SummaryWriter
from util.config import cfg
from util.dist import get_rank, is_primary
from util.log import create_logger
from util.utils_scheduler import adjust_learning_rate, cosine_lr_after_step


def init():
    os.makedirs(cfg.exp_path, exist_ok=True)
    # log the config
    global logger
    logger = create_logger()
    logger.info(cfg)
    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)


def train_one_epoch(epoch, train_loader, model, criterion, optimizer, scaler):

    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()

    net_device = next(model.parameters()).device

    num_iter = len(train_loader)
    max_iter = cfg.epochs * num_iter

    start_time = time.time()
    check_time = time.time()

    for iteration, batch_input in enumerate(train_loader):
        data_time.update(time.time() - check_time)
        torch.cuda.empty_cache()
        current_iter = (epoch - 1) * num_iter + iteration + 1
        remain_iter = max_iter - current_iter

        if epoch > cfg.prepare_epochs:
            curr_lr = adjust_learning_rate(optimizer, current_iter / max_iter, cfg.epochs)
        else:
            curr_lr = cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.prepare_epochs, cfg.epochs)

        for key in batch_input:
            if torch.is_tensor(batch_input[key]):
                batch_input[key] = batch_input[key].to(net_device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(batch_input, epoch)

            if epoch > cfg.prepare_epochs and outputs["mask_predictions"] is None:
                continue

            loss, loss_dict = criterion(outputs, batch_input, epoch)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # time and print

        iter_time.update(time.time() - check_time)
        check_time = time.time()

        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        for k, v in loss_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])
            # writer.add_scalar("Loss/"+k, v[0], iteration)

        if iteration % 10 == 0:
            if epoch <= cfg.prepare_epochs:
                logger.info(
                    "Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Sem loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format(
                        epoch,
                        cfg.epochs,
                        iteration + 1,
                        num_iter,
                        curr_lr,
                        am_dict["loss"].val,
                        am_dict["loss"].avg,
                        am_dict["sem_loss"].val,
                        am_dict["sem_loss"].avg,
                        mem_mb,
                        iter_time.val,
                        remain_time=remain_time,
                    )
                )
            else:
                logger.info(
                    "Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Cls loss: {:.4f}({:.4f}) | Dice loss: {:.4f}({:.4f}) | Focal loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format(
                        epoch,
                        cfg.epochs,
                        iteration + 1,
                        num_iter,
                        curr_lr,
                        am_dict["loss"].val,
                        am_dict["loss"].avg,
                        am_dict["cls_loss"].val,
                        am_dict["cls_loss"].avg,
                        am_dict["dice_loss"].val,
                        am_dict["dice_loss"].avg,
                        am_dict["focal_loss"].val,
                        am_dict["focal_loss"].avg,
                        mem_mb,
                        iter_time.val,
                        remain_time=remain_time,
                    )
                )
                # logger.info("Epoch: {}/{}, iter: {}/{} | lr: {:.6f} | loss: {:.4f}({:.4f}) | Sem loss: {:.4f}({:.4f}) | Cls loss: {:.4f}({:.4f}) | Dice loss: {:.4f}({:.4f}) | Focal loss: {:.4f}({:.4f}) | Mem: {:.2f} | iter_t: {:.2f} | remain_t: {remain_time}\n".format
                #     (epoch, cfg.epochs, iteration + 1, num_iter, curr_lr, am_dict['loss'].val, am_dict['loss'].avg,\
                #     am_dict['sem_loss'].val, am_dict['sem_loss'].avg,
                #     am_dict['cls_loss'].val, am_dict['cls_loss'].avg,\
                #     am_dict['dice_loss'].val, am_dict['dice_loss'].avg, am_dict['focal_loss'].val, am_dict['focal_loss'].avg,
                #     mem_mb,
                #     iter_time.val, remain_time=remain_time))

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


def main():
    # if cfg.ngpus > 1:
    #     init_distributed(
    #         local_rank,
    #         global_rank=local_rank,
    #         world_size=cfg.ngpus,
    #         dist_url=cfg.dist_url,
    #         dist_backend="nccl",
    #     )

    # if is_primary():
    init()

    torch.cuda.set_device(0)
    np.random.seed(cfg.manual_seed + get_rank())
    torch.manual_seed(cfg.manual_seed + get_rank())
    torch.cuda.manual_seed_all(cfg.manual_seed + get_rank())

    # model
    logger.info("=> creating model ...")
    model = GeoFormer()
    model = model.cuda(0)

    # if is_primary():
    logger.info("# training parameters: {}".format(sum([x.nelement() for x in model.parameters() if x.requires_grad])))

    # if is_distributed():
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], find_unused_parameters=True
    #     )

    criterion = InstSetCriterion()
    criterion = criterion.cuda()

    if cfg.optim == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    start_epoch = -1
    if cfg.pretrain:
        if is_primary():
            logger.info("=> loading checkpoint '{}'".format(cfg.pretrain))
        loaded = torch.load(cfg.pretrain, map_location=torch.device("cpu"))["state_dict"]
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(loaded, prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)

    if cfg.resume:
        checkpoint_fn = cfg.resume
        if os.path.isfile(checkpoint_fn):
            if is_primary():
                logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
            start_epoch = state["epoch"] + 1
            # curr_iter = 16000
            # start_iter = 16000
            model_state_dict = model.state_dict()
            loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model.load_state_dict(model_state_dict)
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    dataset = InstDataset(split_set="train")
    train_loader = dataset.trainLoader()

    # if is_primary():
    logger.info(f"Training classes: {dataset.SEMANTIC_LABELS}")
    logger.info("Training samples: {}".format(len(dataset.file_names)))

    if start_epoch == -1:
        start_epoch = 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_one_epoch(epoch, train_loader, model, criterion, optimizer, scaler)


if __name__ == "__main__":
    main()
    # try:
    #     set_start_method("spawn")
    # except RuntimeError:
    #     pass

    # world_size = cfg.ngpus
    # if world_size == 1:
    #     main(local_rank=0)
    # else:
    #     torch.multiprocessing.spawn(main, nprocs=world_size)
