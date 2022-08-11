"""
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
"""

import glob
import json
import multiprocessing as mp
import os

import numpy as np
import plyfile
import scannet_util


DATA_ROOT = "data"
DATASET = "scannetv2"
SAVE_FOLDER = "scenes"

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


files = sorted(glob.glob("data/scannetv2/raws/*_vh_clean_2.ply"))

files2 = sorted(glob.glob("data/scannetv2/raws/*_vh_clean_2.labels.ply"))
files3 = sorted(glob.glob("data/scannetv2/raws/*_vh_clean_2.0.010000.segs.json"))
files4 = sorted(glob.glob("data/scannetv2/raws/*[0-9].aggregation.json"))
assert len(files) == len(files2)
assert len(files) == len(files3)
assert len(files) == len(files4), "{} {}".format(len(files), len(files4))


def f(fn):
    os.makedirs(os.path.join("data/scannetv2", SAVE_FOLDER), exist_ok=True)
    fn2 = fn[:-3] + "labels.ply"
    fn3 = fn[:-15] + "_vh_clean_2.0.010000.segs.json"
    fn4 = fn[:-15] + ".aggregation.json"
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]["label"])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d["segIndices"]
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d["segGroups"]:
            if (
                scannet_util.g_raw2scannetv2[x["label"]] != "wall"
                and scannet_util.g_raw2scannetv2[x["label"]] != "floor"
            ):
                instance_segids.append(x["segments"])
                labels.append(x["label"])
                assert x["label"] in scannet_util.g_raw2scannetv2.keys()
    if (
        fn == "val/scene0217_00_vh_clean_2.ply"
        and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]
    ):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)):
        check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert len(np.unique(sem_labels[pointids])) == 1

    # torch.save((coords, colors, sem_labels, instance_labels), fn[:-15]+'_inst_nostuff.pth')
    # print('Saving to ' + fn[:-15]+'_inst_nostuff.pth')

    sem_labels = np.expand_dims(sem_labels, -1)
    instance_labels = np.expand_dims(instance_labels, -1)
    data_to_save = np.concatenate((coords, colors, sem_labels, instance_labels), 1)
    file_name = fn[-27:-15]
    save_path = os.path.join("data/scannetv2", SAVE_FOLDER, file_name + ".npy")
    np.save(save_path, data_to_save)


# for fn in files:
#     f(fn)

p = mp.Pool(processes=mp.cpu_count())

p.map(f, files)
p.close()
p.join()
