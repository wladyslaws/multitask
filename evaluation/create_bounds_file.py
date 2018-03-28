import json
import numpy as np
from detail import mask
import sys
filename = sys.argv[1]

results = []
data = json.load(open(filename))
for elem in data:
    temp = {}
    temp['name'] = elem['img'][0]
    bound = mask.decode(elem['bound']).tolist()
    temp['mask'] = bound
    results.append(temp)

json.dump(results, open("boundary.json", "w"))
