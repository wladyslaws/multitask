from detaileval_segm import partsSegEvalClass
from detail import Detail
phase='trainval'
det = Detail("../data/datasets/VOC/VOCdevkit/VOC2010/trainval_withkeypoints.json","",phase)
evaluator = partsSegEvalClass(det)
evaluator.load_img_names("../data/datasets/VOC/VOCdevkit/VOC2010/ImageSets/Segmentation/val_list.txt")
evaluator.loadJSON("parts.json")
evaluator.evaluate()
print evaluator.results
