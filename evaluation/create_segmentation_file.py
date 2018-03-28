import json
import numpy as np
from detail import mask
import sys

filename = sys.argv[1]

results = []
data = json.load(open(filename))
for elem in data:
    name = elem['img'][0]
    name = int(name[:4]+name[5:])
    for segm in elem['segm']:
        if segm['id'] == 0:
            continue
        new_segm = {}
        new_segm['parts'] = []
        new_segm['image_id'] = name
        new_segm['category_id'] = segm['id']
        new_segm['segmentation'] = segm['mask']
        results.append(new_segm)    

json.dump({'annos_segmentation': results}, open("segm.json", "w"))
