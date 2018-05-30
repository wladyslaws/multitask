## In order to run this:

please copy VOC2010 photos to their catalogs:
- `VOC2010` to `data/datasets/VOC/VOCdevkit/VOC2010/JPEGImages/`
- `trainval_withkeypoints` to `data/datasets/VOC/VOCdevkit/VOC2010/`

and launch:
> `$python train_ssd500.py`

in order to run.

If you do not change photo sets you can from now on go with:
> `$python train_ssd500.py --load_cache True`

to omit a long step of caching.

Every 100 epochs (number of epochs can be changed with option `-e` or `--epochs`) we perform validation step. The validation step creates data for evaluation in the file `TIME-YYYYMMDD-HHmmss.json`.

If you now switch directory to `evaluation` you can evaluate your results here. For example if you want to evaluate segmentation just go:

>`$python create_segmentation_file.py (path to data TIME json file)`

>`$python script_segmentations.py`
