## Traing and Testing

1\) Pretrain GeoFormer with training classes

```
python3 train.py --config config/geoformer_scannet.yaml --output_path OUTPUT_PATH
```

2\) Episodic Training

```
python3 train_fs.py --config config/geoformer_fs_scannet.yaml --output_path OUTPUT_PATH --pretrain PATH_TO_PRETRAIN_WEIGHT
```

3\) Inference and Evaluation

Test the pretrain model (eval on base classes)

```
python test.py --config config/test_geoformer_scannet.yaml  --output_path OUTPUT_PATH --resume PATH_TO_WEIGHT
```

Test GeoFormer in fewshot setup

```
python test_fs.py --config config/test_geoformer_fs_scannet.yaml --output_path OUTPUT_PATH --resume PATH_TO_WEIGHT
```
