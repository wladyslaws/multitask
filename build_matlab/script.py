import os
import json
import numpy as np
from detail import mask
from progressbar import ProgressBar

f = open("../data/datasets/VOC/VOCdevkit/VOC2010/ImageSets/Segmentation/train_list.txt")
all = f.read().split("\n")[:-1]
all = [elem[:4] + "_" + elem[4:] for elem in all]
result = {}

av_files = [elem.split(".")[0] for elem in os.listdir("dataset/cls/")]
int_files = set.intersection(set(av_files), set(all))


from oct2py import Oct2Py
oc = Oct2Py()
pbar = ProgressBar(maxval=len(int_files)+1).start()
j = 0
for _file in int_files:
    j = j+1
    pbar.update(j)
    out = oc.load("dataset/cls/" + _file + ".mat")
    img = {}
    for i in range(20):
        arr = out['GTcls']['Boundaries'][i].toarray()
        temp = mask.encode(arr.astype(np.uint8))
        img[i] = temp
    result[_file] = img

f = open("result.json", "w")
json.dump(result, f)
pbar.finish()

