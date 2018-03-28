import json
import numpy as np
from detail import mask
import sys
from detaileval_segm import catSegEvalClass
from detaileval_segm import partsSegEvalClass
from detail import Detail
from detail.detaileval_edge import edgeDetectionEval
from detail.detaileval_edge import match, bwmorph, f1_score
#from detection_evaluation_toolset import eval_detection_voc

DATA_ROOT = '/home/zbigniew/'

def get_bboxes(data):
    ids = ids = np.array([2,23,25,31,34,45,59,65,72,98,397,113,207,258,284,308,347,368,416,427,9,18,22,33,44,46,80,85,104,115,144,159,162,220,232,259,260,105,355,295,326,349,19,415,424,440,454,458])
    result = {}
    labels = {}
    for elem in data['annos_segmentation']:
        image_id = elem['image_id']
        label_id = elem['category_id']
        if not image_id in result.keys():
            result[image_id] = []
            labels[image_id] = []
        if not label_id in ids:
            continue
        bbox = elem['bbox']
        result[image_id].append([bbox[0], bbox[1], bbox[0] + bbox[2],  bbox[1] + bbox[3]])
        label = np.where(ids == label_id)[0][0]
        labels[image_id].append(label)
    for key in result.keys():
        result[key] = np.array(result[key])
        labels[key] = np.array(labels[key])
    return (result, labels)

def im2shape_create(data):
        dat = data['images']
        im2shape = {}
        for elem in dat:
            if not elem['image_id'] in im2shape.keys():
                im2shape[elem['image_id']] = (elem['height'], elem['width'])
        return im2shape

filename = sys.argv[1]

phase='trainval'
det = Detail(DATA_ROOT + "data/datasets/VOC/VOCdevkit/VOC2010/trainval_withkeypoints.json","",phase)

results = []
data = json.load(open(filename))
im2shape = im2shape_create(det.data)
gt = json.load(open(DATA_ROOT + "data/datasets/VOC/VOCdevkit/VOC2010/result.json"))
pred = data
names = list(pred.keys())
tps = np.zeros(shape = 20)
fps = np.zeros(shape = 20)
fns = np.zeros(shape = 20)

for name in pred:
    elem = pred[name]
    w,h = im2shape[int(name)]
    gt_elem = gt[name[:4]+"_"+name[4:]]
    for i in range(20):
        mask_pred = bwmorph(mask.decode(elem['sem_bound'][i]))[:w,:h]
        mask_gt = bwmorph(mask.decode(gt_elem[str(i)]))
        tp, fp, fn, _ = match(mask_pred, mask_gt, 100, 0.001)
        tps[i] += tp
        fps[i] += fp
        fns[i] += fn

sem_boundary_results = [f1_score(tps[i], fps[i], fns[i]) for i in range(20)]


for name in data:
    elem = data[name]
    name = int(name)
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

results = []
for name in data:
    temp = {}
    elem = data[name]
    #name = elem['img'][0]
    temp['image_id'] = int(name)
    temp['parts'] = []
    for part in elem['parts']:
        new_part = {}
        new_part['part_id'] = part['part_id']
        new_part['segmentation'] = part['mask']
        temp['parts'].append(new_part)    
    results.append(temp)

json.dump({'annos_segmentation': results}, open("parts.json", "w"))

results = []
for name in data:
    temp = {}
    w,h = im2shape[int(name)]
    elem = data[name]
    temp['name'] = name[:4] + "_" + name[4:]
    bound = mask.decode(elem['bound'])[:w,:h].tolist()
    temp['mask'] = bound
    results.append(temp)

json.dump(results, open("boundary.json", "w"))

evaluator = catSegEvalClass(det)
evaluator.load_img_names(DATA_ROOT + "data/datasets/VOC/VOCdevkit/VOC2010/ImageSets/Segmentation/val_list.txt")
evaluator.loadJSON("segm.json")
evaluator.evaluate()
segmentation_results = evaluator.results

evaluator = partsSegEvalClass(det)
evaluator.load_img_names(DATA_ROOT + "data/datasets/VOC/VOCdevkit/VOC2010/ImageSets/Segmentation/val_list.txt")
evaluator.loadJSON("parts.json")
evaluator.evaluate()
parts_results = evaluator.results

evaluator = edgeDetectionEval(det)
evaluator.loadJSON("boundary.json")
evaluator.evaluate()
boundary_results = evaluator.result

detections, labels = get_bboxes(det.data)
prediction = data
gt_detections = []
gt_labels = []
gt_dificults = []
pred_detections = []
pred_labels = []
pred_dificults = []

for name in prediction.keys():
    gt_detections.append(np.array(detections[int(name)]))
    gt_labels.append(np.array(labels[int(name)]))
    pred_detections.append(np.array(prediction[name]['bbox']))
    pred_labels.append(np.array(prediction[name]['labels']))
    pred_dificults.append(np.array(prediction[name]['probs']))

#res = eval_detection_voc(pred_detections, pred_labels, pred_dificults, gt_detections, gt_labels)


print("Segmentation results: " + str(segmentation_results))
print("Parts results: " + str(parts_results))
print("Boundary results: " + str(boundary_results))
print("Semantic boundary results: " + str(sem_boundary_results))
#print("Detection results: " + str(res))
#print (segmentation_results, parts_results, boundary_results, sem_boundary_results)

