import glob
import os
import pickle
import sys

import numpy as np


sys.path.append("../")


# TRAINING_SEMANTIC_LABELS = [2,3,4,7,9,11,12,13,18]

FOLD0 = [2, 3, 4, 7, 9, 11, 12, 13, 18]
FOLD1 = [5, 6, 8, 10, 14, 15, 16, 17, 19]
FOLD2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

FOLD = {
    0: FOLD0,
    1: FOLD1,
}

FOLD0_NAME = ["cabinet", "bed", "chair", "door", "bookshelf", "counter", "desk", "curtain", "bathtub"]
FOLD1_NAME = [
    "otherfurniture",
    "picture",
    "refridgerator",
    "shower curtain",
    "sink",
    "sofa",
    "table",
    "toilet",
    "window",
]

FOLD_NAME = {
    0: FOLD0_NAME,
    1: FOLD1_NAME,
}

BENCHMARK_SEMANTIC_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


class ScanNetDataset(object):
    def __init__(self, cvfold, data_path):
        self.data_path = data_path
        self.classes = 20
        # self.class2type = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk', 6:'bed', 7:'bookshelf',
        #                    8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet', 12:'curtain', 13:'counter', 14:'door',
        #                    15:'window', 16:'shower curtain', 17:'refridgerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
        class_names = open(os.path.join(self.data_path, "scannet_classnames.txt")).readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in FOLD1]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in FOLD0]
        else:
            raise NotImplementedError("Unknown cvfold (%s). [Options: 0,1]" % cvfold)

        all_classes = [i for i in range(0, self.classes)]
        self.train_classes = [c for c in all_classes if (c not in self.test_classes and c not in [0, 1])]

        self.class2scans_blocks = self.get_class2scans()
        self.class2scans_scenes = self.get_class2scans(block=False)

        self.class2instances = self.get_class2instances()

        self.all_scenes = []
        for file in glob.glob(os.path.join(self.data_path, "scenes", "*.npy")):
            self.all_scenes.append(os.path.basename(file)[:-4])

    def get_class2scans(self, block=True):
        folder_name = "blocks" if block else "scenes"
        class2scans_file = os.path.join(self.data_path, folder_name, "class2scans.pkl")

        if os.path.exists(class2scans_file):
            # load class2scans (dictionary)
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = 0.05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k: [] for k in range(self.classes)}

            for file in glob.glob(os.path.join(self.data_path, folder_name, "*.npy")):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                classes = np.unique(labels)
                print("{0} | shape: {1} | classes: {2}".format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    if class_id == -100:
                        continue
                    # if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print("==== class to scans mapping is done ====")
            for class_id in range(self.classes):
                print(
                    "\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}".format(
                        class_id, min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])
                    )
                )

            with open(class2scans_file, "wb") as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans

    def get_class2instances(self, block=False):
        folder_name = "blocks" if block else "scenes"
        class2instances_file = os.path.join(self.data_path, "class2instances.pkl")

        if os.path.exists(class2instances_file):
            # load class2scans (dictionary)
            with open(class2instances_file, "rb") as f:
                class2instances = pickle.load(f)
        else:
            min_ratio = 0.002  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2instances = {k: [] for k in range(self.classes)}

            for file in sorted(glob.glob(os.path.join(self.data_path, folder_name, "*.npy"))):
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
                    # print("\t Scan {0} | num point {1}".format(scan_name, num_points))
                    if num_points > threshold and class_id != -100:
                        print(
                            "\t class: {0}| num point {1}| instance is {2}".format(class_id, num_points, instance_id)
                        )
                        class2instances[class_id].append([scan_name, instance_id])

            print("==== class to instances mapping is done ====")
            for class_id in range(self.classes):
                print(
                    "\t class_id: {0} | class_name: {1} | num of instances: {2}".format(
                        class_id, self.class2type[class_id], len(class2instances[class_id])
                    )
                )

            with open(class2instances_file, "wb") as f:
                pickle.dump(class2instances, f, pickle.HIGHEST_PROTOCOL)
        return class2instances
