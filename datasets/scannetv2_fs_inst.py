"""
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
"""

import math
import os
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import DataLoader

import pickle
import random

from lib.pointgroup_ops.functions import pointgroup_ops
from util.config import cfg

from .scannetv2 import FOLD


class FSInstDataset:
    def __init__(self, split_set="train"):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset

        self.batch_size = cfg.batch_size

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        self.split_set = split_set

        split_filenames = os.path.join(self.data_root, self.dataset, f"scannetv2_{split_set}.txt")
        with open(split_filenames, "r") as f:
            self.scan_names = f.read().splitlines()

        all_file_names = os.listdir(os.path.join(self.data_root, self.dataset, "scenes"))
        self.file_names = [
            os.path.join(self.data_root, self.dataset, "scenes", f)
            for f in all_file_names
            if f.split(".")[0] in self.scan_names
        ]
        self.file_names = sorted(self.file_names)

        self.SEMANTIC_LABELS = FOLD[cfg.cvfold]
        self.SEMANTIC_LABELS_MAP = {val: (idx + 4) for idx, val in enumerate(self.SEMANTIC_LABELS)}

        class2scans_file = os.path.join(self.data_root, self.dataset, "class2scans.pkl")
        with open(class2scans_file, "rb") as f:
            self.class2scans_scenes = pickle.load(f)

        class2instances_file = os.path.join(self.data_root, self.dataset, "class2instances.pkl")
        with open(class2instances_file, "rb") as f:
            self.class2instances = pickle.load(f)

    def __len__(self):
        return len(self.file_names)

    def my_worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

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
            collate_fn=self.trainMergeFS,
        )
        return dataloader

    def testLoader(self):

        self.test_names = [os.path.basename(i).split(".")[0][:12] for i in self.file_names]
        self.test_combs = self.get_test_comb()
        test_set = list(np.arange(len(self.test_names)))
        dataloader = DataLoader(
            test_set,
            batch_size=1,
            collate_fn=self.testMergeFS,
            num_workers=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        return dataloader

    def get_test_comb(self):
        test_combs_file = os.path.join(
            self.data_root, self.dataset, "test_combinations_fold" + str(cfg.cvfold) + ".pkl"
        )

        if os.path.exists(test_combs_file):
            # load class2scans (dictionary)
            print("Load test combination: ", test_combs_file)
            with open(test_combs_file, "rb") as f:
                test_combs = pickle.load(f)
        else:
            test_combs = {k: {} for k in self.test_names}
            for i, file_name in enumerate(self.test_names):
                data = np.load(os.path.join(self.data_root, self.dataset, "scenes", "%s.npy" % file_name))
                label = data[:, 6].astype(np.int)
                unique_label = np.unique(label)
                active_label = []
                for l in unique_label:
                    if l == -100:
                        continue

                    if l in FOLD[cfg.cvfold]:
                        active_label.append(l)

                test_combs[file_name]["active_label"] = active_label
                if len(active_label) == 0:
                    continue

                for l in active_label:
                    support_tuple = random.choice(self.class2instances[l])
                    test_combs[file_name][l] = support_tuple

            with open(test_combs_file, "wb") as f:
                pickle.dump(test_combs, f, pickle.HIGHEST_PROTOCOL)

        print("len test combs:", len(list(test_combs.keys())))
        return test_combs

    def genSupportLoader(self):
        self.support_set = self.get_support_set(k_shot=cfg.k_shot)

    def get_support_set(self, k_shot):
        support_sets_dir = os.path.join(self.data_root, self.dataset, "support_sets")
        os.makedirs(support_sets_dir, exist_ok=True)

        support_set_file = os.path.join(
            self.data_root,
            self.dataset,
            "support_sets",
            cfg.type_support + str(cfg.cvfold) + "_" + str(k_shot) + "shot_10sets.pkl",
        )
        if os.path.exists(support_set_file):
            print(f"Load support sets: {support_set_file}")
            with open(support_set_file, "rb") as f:
                support_sets = pickle.load(f)
        else:
            print("Generate new support sets")
            support_sets = []
            for subset in range(0, cfg.run_num):
                random.seed(10 * subset)
                support_set = {cls: [] for cls in self.SEMANTIC_LABELS}
                for cls in self.SEMANTIC_LABELS:
                    for i in range(k_shot):
                        while True:
                            support_tuple = random.choice(self.class2instances[cls])
                            support_scene_name, support_instance_id = support_tuple[0], support_tuple[1]

                            (
                                support_xyz_middle,
                                support_xyz_scaled,
                                support_rgb,
                                support_label,
                                support_instance_label,
                            ) = self.load_single(support_scene_name, aug=False, permutate=False, val=True)
                            support_mask = (support_instance_label == support_instance_id).astype(int)

                            if np.count_nonzero(support_mask) >= 1000:
                                print("Pick {}, {}".format(support_scene_name, support_instance_id))
                                break
                        support_set[cls].append(support_tuple)

                support_sets.append(support_set)
            with open(support_set_file, "wb") as f:
                pickle.dump(support_sets, f, pickle.HIGHEST_PROTOCOL)
        return support_sets

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

    def get_sphere_inst(self, support_xyz_middle, support_instance_label, support_instance_id, scale_factor=2):
        scale_factor = scale_factor / 2
        if scale_factor == -1:
            return np.arange(support_xyz_middle.shape[0])

        inst_pc = support_xyz_middle[support_instance_label == support_instance_id]

        centroid = np.mean(inst_pc, axis=0)

        max_distance = np.max(np.linalg.norm(inst_pc - centroid, axis=1))
        radius = scale_factor * max_distance
        absolute_distance = np.linalg.norm(support_xyz_middle - centroid, axis=1)
        valid_pc = absolute_distance <= radius
        valid_indices = np.nonzero(valid_pc)
        return valid_indices

    def get_region_inst(self, support_xyz_middle, support_instance_label, support_instance_id, scale_factor=2):
        scale_factor = scale_factor / 2
        if scale_factor == -1:
            return np.arange(support_xyz_middle.shape[0])
        inst_pc = support_xyz_middle[support_instance_label == support_instance_id]
        xmin = np.min(inst_pc[:, 0])
        ymin = np.min(inst_pc[:, 1])
        zmin = np.min(inst_pc[:, 2])
        xmax = np.max(inst_pc[:, 0])
        ymax = np.max(inst_pc[:, 1])
        zmax = np.max(inst_pc[:, 2])
        x_middle = (xmin + xmax) / 2
        y_middle = (ymin + ymax) / 2
        z_middle = (zmin + zmax) / 2
        x_size = xmax - xmin + 0.1
        y_size = ymax - ymin + 0.1
        z_size = zmax - zmin + 0.1

        x_lower = x_middle - x_size * scale_factor
        x_upper = x_middle + x_size * scale_factor
        y_lower = y_middle - y_size * scale_factor
        y_upper = y_middle + y_size * scale_factor
        z_lower = z_middle - z_size * scale_factor
        z_upper = z_middle + z_size * scale_factor

        valid_pc = (
            (support_xyz_middle[:, 0] <= x_upper)
            & (support_xyz_middle[:, 0] >= x_lower)
            & (support_xyz_middle[:, 1] <= y_upper)
            & (support_xyz_middle[:, 1] >= y_lower)
            & (support_xyz_middle[:, 2] <= z_upper)
            & (support_xyz_middle[:, 2] >= z_lower)
        )

        valid_indices = np.nonzero(valid_pc)
        return valid_indices

    def load_single(self, scene_name, aug=True, permutate=True, val=False, support=False):
        data = np.load(os.path.join(self.data_root, self.dataset, "scenes", "%s.npy" % scene_name))

        xyz_origin = data[:, :3]
        rgb = data[:, 3:6]
        label = data[:, 6].astype(np.int)
        instance_label = data[:, 7].astype(np.int)

        # jitter / flip x / rotation
        if aug:
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)
        else:
            xyz_middle = xyz_origin

        # scale
        xyz = xyz_middle * self.scale

        # elastic
        if aug:
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        # offset
        xyz -= xyz.min(0)

        if not val and not support:
            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = instance_label[valid_idxs]

        return xyz_middle, xyz, rgb, label, instance_label

    def load_single_block(self, scene_name, instance_id, aug=True, permutate=True, val=False):
        data = np.load(os.path.join(self.data_root, self.dataset, "scenes", "%s.npy" % scene_name))

        if permutate:
            random_idx = np.random.permutation(data.shape[0])
            data = data[random_idx]

        xyz_origin = data[:, 0:3]
        rgb = data[:, 3:6]
        label = data[:, 6].astype(np.int)
        instance_label = data[:, 7].astype(np.int)

        # jitter / flip x / rotation
        if aug:
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)
        else:
            xyz_middle = xyz_origin

        # valid_indices = self.get_sphere_inst(xyz_middle, instance_label, instance_id, scale_factor=8)
        valid_indices = self.get_region_inst(xyz_middle, instance_label, instance_id, scale_factor=1)
        # valid_indices = np.nonzero(instance_label == instance_id)
        xyz_middle = xyz_middle[valid_indices]
        rgb = rgb[valid_indices]
        label = label[valid_indices]
        instance_label = instance_label[valid_indices]

        xyz = xyz_middle * self.scale
        xyz -= xyz.min(0)
        return xyz_middle, xyz, rgb, label, instance_label

    def trainMergeFS(self, ids):
        support_locs = []
        support_locs_float = []
        support_feats = []
        support_masks = []
        support_batch_offsets = [0]
        support_pc_mins = []
        support_pc_maxs = []

        query_locs = []
        query_locs_float = []
        query_feats = []
        query_labels = []
        query_instance_labels = []
        # query_instance_infos = []  # (N, 9)
        query_instance_pointnum = []  # (total_nInst), int
        query_batch_offsets = [0]
        query_total_inst_num = 0
        query_pc_mins = []
        query_pc_maxs = []
        scene_infos = []

        total_inst_num = 0
        for idx, id in enumerate(ids):
            sampled_class = random.choice(self.SEMANTIC_LABELS)

            query_scene_name = random.choice(self.class2scans_scenes[sampled_class])

            query_xyz_middle, query_xyz_scaled, query_rgb, query_label, query_instance_label = self.load_single(
                query_scene_name, aug=True, permutate=True, val=False
            )
            query_label = query_label == sampled_class
            query_instance_label[(query_label == 0).nonzero()] = -100

            query_instance_label = self.getCroppedInstLabel(query_instance_label, valid_idxs=None)
            # get instance information
            query_inst_num, inst_infos = self.getInstanceInfo(query_xyz_middle, query_instance_label.astype(np.int32))
            query_inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            query_instance_label[np.where(query_instance_label != -100)] += total_inst_num
            query_total_inst_num += query_inst_num

            query_batch_offsets.append(query_batch_offsets[-1] + query_xyz_scaled.shape[0])
            query_locs.append(
                torch.cat(
                    [
                        torch.LongTensor(query_xyz_scaled.shape[0], 1).fill_(idx),
                        torch.from_numpy(query_xyz_scaled).long(),
                    ],
                    1,
                )
            )
            query_locs_float.append(torch.from_numpy(query_xyz_middle))
            query_feats.append(torch.from_numpy(query_rgb))
            query_labels.append(torch.from_numpy(query_label))
            query_instance_labels.append(torch.from_numpy(query_instance_label))
            # query_instance_infos.append(torch.from_numpy(query_inst_infos))
            query_instance_pointnum.extend(query_inst_pointnum)

            query_pc_mins.append(torch.from_numpy(query_xyz_middle.min(axis=0)))
            query_pc_maxs.append(torch.from_numpy(query_xyz_middle.max(axis=0)))

            # ANCHOR Sampling support
            while True:
                support_tuple = random.choice(self.class2instances[sampled_class])
                support_scene_name, support_instance_id = support_tuple[0], support_tuple[1]

                (
                    support_xyz_middle,
                    support_xyz_scaled,
                    support_rgb,
                    support_label,
                    support_instance_label,
                ) = self.load_single(support_scene_name, aug=False, permutate=False, val=False, support=True)

                support_mask = support_instance_label == support_instance_id

                if np.count_nonzero(support_label) > 100:
                    break

            support_batch_offsets.append(support_batch_offsets[-1] + support_xyz_scaled.shape[0])
            support_locs.append(
                torch.cat(
                    [
                        torch.LongTensor(support_xyz_scaled.shape[0], 1).fill_(idx),
                        torch.from_numpy(support_xyz_scaled).long(),
                    ],
                    1,
                )
            )
            support_locs_float.append(torch.from_numpy(support_xyz_middle))
            support_feats.append(torch.from_numpy(support_rgb))
            support_masks.append(torch.from_numpy(support_mask))

            support_pc_mins.append(torch.from_numpy(support_xyz_middle.min(axis=0)))
            support_pc_maxs.append(torch.from_numpy(support_xyz_middle.max(axis=0)))

            scene_infos.append(
                {
                    "sampled_class": sampled_class,
                    "query_scene": query_scene_name,
                    "support_scene": support_scene_name,
                    "support_instance_id": support_instance_id,
                }
            )

        support_batch_offsets = torch.tensor(support_batch_offsets, dtype=torch.int)  # int (B+1)
        support_locs = torch.cat(support_locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        support_locs_float = torch.cat(support_locs_float, 0).to(torch.float32)  # float (N, 3)
        support_feats = torch.cat(support_feats, 0)  # float (N, C)
        support_masks = torch.cat(support_masks, 0).long()  # long (N)
        support_spatial_shape = np.clip((support_locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

        # voxelize
        support_voxel_locs, support_p2v_map, support_v2p_map = pointgroup_ops.voxelization_idx(
            support_locs, self.batch_size, self.mode
        )

        support_pc_mins = torch.stack(support_pc_mins).float()
        support_pc_maxs = torch.stack(support_pc_maxs).float()

        support_dict = {
            "voxel_locs": support_voxel_locs,
            "p2v_map": support_p2v_map,
            "v2p_map": support_v2p_map,
            "locs": support_locs,
            "locs_float": support_locs_float,
            "feats": support_feats,
            "support_masks": support_masks,
            "spatial_shape": support_spatial_shape,
            "batch_offsets": support_batch_offsets,
            "pc_mins": support_pc_mins,
            "pc_maxs": support_pc_maxs,
        }

        query_batch_offsets = torch.tensor(query_batch_offsets, dtype=torch.int)  # int (B+1)
        query_locs = torch.cat(query_locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        query_locs_float = torch.cat(query_locs_float, 0).to(torch.float32)  # float (N, 3)
        query_feats = torch.cat(query_feats, 0)  # float (N, C)
        query_labels = torch.cat(query_labels, 0).long()
        query_instance_labels = torch.cat(query_instance_labels, 0).long()  # long (N)
        # query_instance_infos = torch.cat(query_instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        query_instance_pointnum = torch.tensor(
            query_instance_pointnum, dtype=torch.int
        )  # int (total_nInst)                     # long (N)
        query_spatial_shape = np.clip((query_locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

        # voxelize
        query_voxel_locs, query_p2v_map, query_v2p_map = pointgroup_ops.voxelization_idx(
            query_locs, self.batch_size, self.mode
        )
        query_pc_mins = torch.stack(query_pc_mins).float()
        query_pc_maxs = torch.stack(query_pc_maxs).float()
        query_dict = {
            "voxel_locs": query_voxel_locs,
            "p2v_map": query_p2v_map,
            "v2p_map": query_v2p_map,
            "locs": query_locs,
            "locs_float": query_locs_float,
            "feats": query_feats,
            "labels": query_labels,
            "instance_labels": query_instance_labels,
            "instance_pointnum": query_instance_pointnum,
            "spatial_shape": query_spatial_shape,
            "batch_offsets": query_batch_offsets,
            "pc_mins": query_pc_mins,
            "pc_maxs": query_pc_maxs,
        }

        return support_dict, query_dict, scene_infos

    def testMergeFS(self, ids):
        index = ids[0]  # batch size 1

        query_scene_name = self.test_names[index]

        test_comb = self.test_combs[query_scene_name]

        # NOTE test scene does not contain any test categories
        if len(test_comb["active_label"]) == 0:
            return False, {}, {}, {}

        scene_infos = {"query_scene": query_scene_name, "active_label": test_comb["active_label"]}
        query_xyz_middle, query_xyz_scaled, query_rgb, query_label, query_instance_label = self.load_single(
            query_scene_name, aug=False, permutate=False, val=True
        )

        query_locs = torch.cat(
            [torch.LongTensor(query_xyz_scaled.shape[0], 1).fill_(0), torch.from_numpy(query_xyz_scaled).long()], 1
        )
        query_locs_float = torch.from_numpy(query_xyz_middle).to(torch.float32)
        query_feats = torch.from_numpy(query_rgb).to(torch.float32)
        query_batch_offsets = torch.tensor([0, query_xyz_middle.shape[0]], dtype=torch.int)

        query_label = torch.from_numpy(query_label)
        # query_instance_label = torch.from_numpy(query_instance_label)

        query_spatial_shape = np.clip((query_locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

        query_voxel_locs, query_p2v_map, query_v2p_map = pointgroup_ops.voxelization_idx(
            query_locs, self.batch_size, self.mode
        )

        query_pc_min = torch.from_numpy(query_xyz_middle.min(axis=0)).unsqueeze(0)
        query_pc_max = torch.from_numpy(query_xyz_middle.max(axis=0)).unsqueeze(0)

        query_dict = {
            "voxel_locs": query_voxel_locs,
            "p2v_map": query_p2v_map,
            "v2p_map": query_v2p_map,
            "locs": query_locs,
            "locs_float": query_locs_float,
            "feats": query_feats,
            "spatial_shape": query_spatial_shape,
            "batch_offsets": query_batch_offsets,
            "pc_mins": query_pc_min,
            "pc_maxs": query_pc_max,
            "labels": query_label,
        }

        if cfg.fix_support:
            list_support_dicts = [None] * len(test_comb["active_label"])
        else:
            list_support_dicts = []
            for l in test_comb["active_label"]:
                support_tuple = test_comb[l]
                support_scene_name, support_instance_id = support_tuple[0], support_tuple[1]
                scene_infos[l] = support_tuple

                (
                    support_xyz_middle,
                    support_xyz_scaled,
                    support_rgb,
                    support_label,
                    support_instance_label,
                ) = self.load_single_block(
                    support_scene_name, support_instance_id, aug=False, permutate=False, val=True
                )

                support_mask = (support_instance_label == support_instance_id).astype(int)

                support_pc_min = torch.from_numpy(support_xyz_middle.min(axis=0)).unsqueeze(0)
                support_pc_max = torch.from_numpy(support_xyz_middle.max(axis=0)).unsqueeze(0)

                support_batch_offsets = torch.tensor([0, support_xyz_middle.shape[0]], dtype=torch.int)
                support_masks_offset = torch.tensor([0, np.count_nonzero(support_mask)], dtype=torch.int)  # int (B+1)
                support_locs = torch.cat(
                    [
                        torch.LongTensor(support_xyz_scaled.shape[0], 1).fill_(0),
                        torch.from_numpy(support_xyz_scaled).long(),
                    ],
                    1,
                )
                support_locs_float = torch.from_numpy(support_xyz_middle).to(torch.float32)
                support_feats = torch.from_numpy(support_rgb).to(torch.float32)  # float (N, C)
                support_masks = torch.from_numpy(support_mask)
                support_spatial_shape = np.clip(
                    (support_locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale_support[0], None
                )

                # voxelize
                support_voxel_locs, support_p2v_map, support_v2p_map = pointgroup_ops.voxelization_idx(
                    support_locs, 1, self.mode
                )

                support_dict = {
                    "voxel_locs": support_voxel_locs,
                    "p2v_map": support_p2v_map,
                    "v2p_map": support_v2p_map,
                    "locs": support_locs,
                    "locs_float": support_locs_float,
                    "feats": support_feats,
                    "support_masks": support_masks,
                    "spatial_shape": support_spatial_shape,
                    "batch_offsets": support_batch_offsets,
                    "mask_offsets": support_masks_offset,
                    "pc_mins": support_pc_min,
                    "pc_maxs": support_pc_max,
                }
                list_support_dicts.append(support_dict)

        return True, list_support_dicts, query_dict, scene_infos
