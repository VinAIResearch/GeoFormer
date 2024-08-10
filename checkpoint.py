import logging
from collections import OrderedDict

import torch

import errno
import os


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            print("+++" * 5 + "{} not loaded".format(current_keys[idx_new]))
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if loaded_state_dict[key_old].shape != model_state_dict[key].shape:
            if "unet" in key or "input_conv" in key:
                reshaped = loaded_state_dict[key_old].permute(4, 0, 1, 2, 3)
                loaded_state_dict[key_old] = reshaped
            else:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        key,
                        model_state_dict[key].shape,
                        loaded_state_dict[key_old].shape,
                    )
                )
                loaded_state_dict[key_old] = model_state_dict[key]

        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # stripped_state_dict[key.replace(prefix, "")] = value
        stripped_state_dict[key[len(prefix) :]] = value
    return stripped_state_dict


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def checkpoint(
    model,
    optimizer,
    epoch,
    log_dir,
    best_val=None,
    best_val_iter=None,
    postfix=None,
    last=False,
):
    mkdir_p(log_dir)

    if last:
        filename = "checkpoint_last.pth"
    else:
        filename = f"checkpoint_epoch_{epoch}.pth"
    checkpoint_file = log_dir + "/" + filename
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
