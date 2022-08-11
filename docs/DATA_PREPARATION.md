## ScanNet v2 dataset

1\) We prepared a preprocessed version of ScanNetv2 dataset. Download it at [GoogleDrive](https://drive.google.com/file/d/1zV3hV2wBM0sd_zKeD1uCDSu-TCExaJzk/view?usp=sharing)

2\) Extract the downloaded zip file to `geoformer/data` as follows.

```
geoformer
├── data
│   ├── scannetv2
│   │   ├── scenes
│   │   ├── val_gt
│   │   ├── support_sets
│   │   ├── class2instances.pkl
│   │   ├── class2scans.pkl
│   │   ├── test_combinations_fold0
│   │   ├── test_combinations_fold1
│   │   ├── scannetv2_train.txt
│   │   ├── scannetv2_test.txt
│   │   ├── scannetv2_val.txt
...
```
where `scenes` is the preprocessed 3d point cloud data stored in `.npy` format, `val_gt` stores groundtruth instance masks, `support_sets` stores the predefined support sets (used to eval all experiments)

3\) If you cannot download our preprocessed data, you can download the original ScanNetv2 dataset from [ScanNet](http://www.scan-net.org/). Put all the files in `geoformer/data/scannetv2/raws` folder and run two scripts `data/scannetv2/prepare_data_inst.py` and `data/scannetv2/prepare_data_inst_gt.py`.