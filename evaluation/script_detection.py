import numpy as np
import sys
import json
from detail import Detail
from detection_evaluation_toolset import eval_detection_voc

phase='trainval'

def get_bboxes(data):
    ids = np.array([2,259,260,415,324,9,258,144,18,19,22,23,397,25,284,158,31,416,33,34,420,454,295,296,427,
        44,45,46,349,232,308,440,59,189,65,354,424,68,326,72,458,162,207,80,355,85,347,220,159,360,98,187,104,105,366,445,368,113,115])
    result = {}
    labels = {}
    for elem in data['annos_segmentation']:
        image_id = elem['image_id']
        if not image_id in result.keys():
            result[image_id] = []
            labels[image_id] = []
        bbox = elem['bbox']
        result[image_id].append([bbox[0], bbox[1], bbox[0] + bbox[2],  bbox[1] + bbox[3]])
        label_id = elem['category_id']
        label = np.where(ids == label_id)[0][0]
        labels[image_id].append(label)
    for key in result.keys():
        result[key] = np.array(result[key])
        labels[key] = np.array(labels[key])
    return (result, labels)


filename = sys.argv[1]
prediction = json.load(open(filename))
det = Detail("../data/datasets/VOC/VOCdevkit/VOC2010/trainval_withkeypoints.json","",phase)
detections, labels = get_bboxes(det.data)

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

res = eval_detection_voc(pred_detections, pred_labels, pred_dificults, gt_detections, gt_labels)

print(res)
