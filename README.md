In order to run this:
please copy photos of voc2010 and trainval\_withkeypoints to their catalogs (data/datasets/VOC/VOCdevkit/VOC2010/JPEGImages/ and data/datasets/VOC/VOCdevkit/VOC2010/ respectively).
In pytorch\_fcn run python setup.py install
Now go to pytorch-fcn/examples/voc/ and run python train\_ssd500.py -g 0in order to run.
If you do not chcange photo sets you can from now on go with python train\ssd500.py -g 0 --load\_cache True thus ommiting a long step of caching.
every 100 epochs (can be changed easily) we perform validation step. The validation step creates data for evaluation epochx.json where x is number of epoch. The json can be found under pytorch-fcn/examples/voc/logs/Model-(something)/epochx.json
If you now switch directory to evaluation you can evaluate your results here. For example if you want to evaluate segmentation just go:
python create\_segmentation\_file.py (path to epoch)
python script\_segmentations.py
