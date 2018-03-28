import json
import numpy as np
from detail import mask
import sys
filename = sys.argv[1]

results = []
data = json.load(open(filename))
for elem in data:
    temp = {}
    name = elem['img'][0]
    temp['image_id'] = int(name[:4]+name[5:])
    temp['parts'] = []
    for part in elem['parts']:
        new_part = {}
        new_part['part_id'] = part['part_id']
        new_part['segmentation'] = part['mask']
        temp['parts'].append(new_part)    
    results.append(temp)

json.dump({'annos_segmentation': results}, open("parts.json", "w"))
