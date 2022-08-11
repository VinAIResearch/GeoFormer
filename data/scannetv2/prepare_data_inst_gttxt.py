"""
Generate instance groundtruth .txt files (for evaluation)
"""
import os
import numpy as np


semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


if __name__ == "__main__":
    split = "val"
    # files = sorted(glob.glob('{}/scene*_inst_nostuff.pth'.format(split)))
    split_filenames = os.path.join("data/scannetv2/", f"scannetv2_{split}.txt")
    with open(split_filenames, "r") as f:
        scan_names = f.read().splitlines()
    rooms = os.listdir(os.path.join("data/scannetv2/scenes"))
    rooms_files = [os.path.join("data/scannetv2/scenes", f) for f in rooms if f.split(".")[0] in scan_names]

    if not os.path.exists(os.path.join("data/scannetv2/", split + "_gt")):
        os.mkdir(os.path.join("data/scannetv2/", split + "_gt"))

    for i in range(len(rooms_files)):
        # xyz, rgb, label, instance_label = rooms[i]   # label 0~19 -100;  instance_label 0~instance_num-1 -100
        file_path = rooms_files[i]
        scene_name = file_path.split("/")[-1][:12]
        data = np.load(file_path)
        random_idx = np.random.permutation(data.shape[0])
        data = data[random_idx]
        xyz_origin = data[:, 0:3]
        rgb = data[:, 3:6]
        label = data[:, 6].astype(np.int)
        instance_label = data[:, 7].astype(np.int)
        print("{}/{} {}".format(i + 1, len(rooms_files), scene_name))

        instance_label_new = np.zeros(
            instance_label.shape, dtype=np.int32
        )  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if sem_id == -100:
                sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join("data/scannetv2/", split + "_gt", scene_name + ".txt"), instance_label_new, fmt="%d")
