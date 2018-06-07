## In order to run this:

---

Important! Somehow files with names: **2010_001592.jpg, 2010_001606.jpg** and **2008_003252.jpg** lack their respective annotations in trainval_withkeypoints.json data, hence you have to remove them from downloaded JPEGImages folder in order to evade dreadful crash during training. We are working to make this code failsafe in this matter.

Remove mentioned images ids, i.e.  **2010_001592*, **2010_001606**, **2008_003252** from **train_list.txt** and **val_list.txt** files which are a source of the images for the training process.

---

Please copy VOC2010 photos to their catalogs:
- `VOC2010` to `data/datasets/VOC/VOCdevkit/VOC2010/JPEGImages/`
- `trainval_withkeypoints` to `data/datasets/VOC/VOCdevkit/VOC2010/`

and launch:
> `$python train_ssd500.py`

in order to run.

If you do not change photo sets you can from now on go with:
> `$python train_ssd500.py --load_cache True`

to omit a long step of caching. You can change number of epochs with option `-e` or `--epochs`.

At the end of training we perform validation step. The validation step creates data for evaluation in the file `TIME-YYYYMMDD-hhmmss.json`.

If you now switch directory to `evaluation` you can evaluate your results here. For example if you want to evaluate segmentation just go:

>`$python create_segmentation_file.py (path to data TIME json file)`

>`$python script_segmentations.py`
