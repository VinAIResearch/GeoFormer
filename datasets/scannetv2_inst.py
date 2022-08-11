"""
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
"""

import glob
import math
import os
import numpy as np
from torch.utils.data import Dataset as TorchDataset

import pickle

import scipy.interpolate
import scipy.ndimage
import torch
from lib.pointgroup_ops.functions import pointgroup_ops
from torch.utils.data import DataLoader
from util.config import cfg

from .scannetv2 import FOLD


class InstDataset(TorchDataset):
    def __init__(self, split_set="train"):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset

        self.batch_size = cfg.batch_size

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        split_filenames = os.path.join(self.data_root, self.dataset, f"scannetv2_{split_set}.txt")
        with open(split_filenames, "r") as f:
            self.scan_names = f.read().splitlines()

        all_file_names = os.listdir(os.path.join(self.data_root, self.dataset, "scenes"))
        self.file_names = [
            os.path.join(self.data_root, self.dataset, "scenes", f)
            for f in all_file_names
            if f.split(".")[0][:12] in self.scan_names
        ]
        self.file_names = sorted(self.file_names)

        self.SEMANTIC_LABELS = FOLD[cfg.cvfold]
        self.SEMANTIC_LABELS_MAP = {val: (idx + 4) for idx, val in enumerate(self.SEMANTIC_LABELS)}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        return index

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_root, self.dataset, "class2scans.pkl")

        if os.path.exists(class2scans_file):
            # load class2scans (dictionary)
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = 0.05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k: [] for k in FOLD[2]}  # NOTE generate all classes

            for file in glob.glob(os.path.join(self.data_root, self.dataset, "scenes", "*.npy")):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                classes = np.unique(labels)
                print("{0} | shape: {1} | classes: {2}".format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    if class_id == -100 or class_id not in FOLD[2]:
                        continue
                    # if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print("==== class to scans mapping is done ====")
            for class_id in FOLD[2]:
                print(
                    "\t class_id: {0} | min_ratio: {1} | min_pts: {2} | num of scans: {3}".format(
                        class_id, min_ratio, min_pts, len(class2scans[class_id])
                    )
                )

            with open(class2scans_file, "wb") as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans

    def get_class2instances(
        self,
    ):
        class2instances_file = os.path.join(self.data_root, self.dataset, "class2instances.pkl")

        if os.path.exists(class2instances_file):
            # load class2scans (dictionary)
            with open(class2instances_file, "rb") as f:
                class2instances = pickle.load(f)
        else:
            min_ratio = 0.01  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2instances = {k: [] for k in FOLD[2]}

            for file in sorted(glob.glob(os.path.join(self.data_root, self.dataset, "scenes", "*.npy"))):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                instance_labels = data[:, 7].astype(np.int)
                instances = np.unique(instance_labels)
                # print("\t Scan {0} | num instance {1}".format(scan_name, instance_labels.shape))
                for instance_id in instances:
                    if instance_id == -100:
                        continue
                    num_points = np.count_nonzero(instance_labels == instance_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    one_point = (instance_labels == instance_id).nonzero()[0][0]
                    class_id = labels[one_point]
                    if class_id not in FOLD[2]:
                        continue
                    # print("\t Scan {0} | num point {1}".format(scan_name, num_points))
                    if num_points > threshold and class_id != -100:
                        print(
                            "\t class: {0}| num point {1}| instance is {2}".format(class_id, num_points, instance_id)
                        )
                        class2instances[class_id].append([scan_name, instance_id])

            print("==== class to instances mapping is done ====")
            for class_id in FOLD[2]:
                print("\t class_id: {0} |  num of instances: {1}".format(class_id, len(class2instances[class_id])))

            with open(class2instances_file, "wb") as f:
                pickle.dump(class2instances, f, pickle.HIGHEST_PROTOCOL)
        return class2instances

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        """
        instance_info = (
            np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
        )  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            # instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            # instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs=None):
        if valid_idxs is not None:
            instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def trainLoader(self):

        train_set = list(range(len(self.file_names)))
        dataloader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=self.trainMerge,
        )
        return dataloader

    def testLoader(self):

        self.test_names = [os.path.basename(i).split(".")[0][:12] for i in self.file_names]

        test_set = list(np.arange(len(self.test_names)))

        dataloader = DataLoader(
            test_set,
            batch_size=1,
            collate_fn=self.testMerge,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        return dataloader

    def trainMerge(self, inds):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        pc_mins = []
        pc_maxs = []

        total_inst_num = 0
        for i, ind in enumerate(inds):
            file_path = self.file_names[ind]
            data = np.load(file_path)

            xyz_origin = data[:, :3]
            rgb = data[:, 3:6]
            label = data[:, 6].astype(np.int)
            instance_label = data[:, 7].astype(np.int)

            # jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            # scale
            xyz = xyz_middle * self.scale

            # elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            # offset
            xyz -= xyz.min(0)

            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = instance_label[valid_idxs]

            # ANCHOR modify semantic label
            label_2 = np.ones_like(label) * -1
            label_2[(label == 0).nonzero()] = 0  # floor
            label_2[(label == 1).nonzero()] = 1  # wall
            for idx, train_class in enumerate(self.SEMANTIC_LABELS):
                label_2[(label == train_class).nonzero()] = idx + 4
            label_2[(label == -100).nonzero()] = 2  # unannotate
            label_2[(label_2 == -1).nonzero()] = 3  # test candidate

            label = label_2
            instance_label[(label <= 3).nonzero()] = -100

            instance_label = self.getCroppedInstLabel(instance_label)

            # get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

            pc_mins.append(torch.from_numpy(xyz_middle.min(axis=0)))
            pc_maxs.append(torch.from_numpy(xyz_middle.max(axis=0)))

        # merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

        pc_mins = torch.stack(pc_mins).float()
        pc_maxs = torch.stack(pc_maxs).float()

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {
            "locs": locs,
            "voxel_locs": voxel_locs,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "locs_float": locs_float,
            "feats": feats,
            "labels": labels,
            "instance_labels": instance_labels,
            "instance_pointnum": instance_pointnum,
            "instance_infos": instance_infos,
            "id": inds,
            "offsets": batch_offsets,
            "spatial_shape": spatial_shape,
            "pc_mins": pc_mins,
            "pc_maxs": pc_maxs,
        }

    def testMerge(self, inds):
        locs = []
        locs_float = []
        feats = []
        test_scene_name = []

        batch_offsets = [0]

        pc_mins = []
        pc_maxs = []

        for i, ind in enumerate(inds):
            file_path = self.file_names[ind]
            data = np.load(file_path)

            xyz_origin = data[:, :3]
            rgb = data[:, 3:6]

            # flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            # scale
            xyz = xyz_middle * self.scale

            # offset
            xyz -= xyz.min(0)

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            test_scene_name.append(self.test_names[ind])

            pc_mins.append(torch.from_numpy(xyz_middle.min(axis=0)))
            pc_maxs.append(torch.from_numpy(xyz_middle.max(axis=0)))

        pc_mins = torch.stack(pc_mins).float()
        pc_maxs = torch.stack(pc_maxs).float()

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {
            "locs": locs,
            "voxel_locs": voxel_locs,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "locs_float": locs_float,
            "feats": feats,
            "id": inds,
            "offsets": batch_offsets,
            "spatial_shape": spatial_shape,
            "test_scene_name": test_scene_name,
            "pc_mins": pc_mins,
            "pc_maxs": pc_maxs,
        }
