#!/usr/bin/env python
import argparse
import datetime
import math
import os
import os.path as osp
import sys
import numpy as np
import pytz
import json
import itertools
import sys
import collections
import pickle
import PIL.Image
from detail import Detail, mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import multitask_utils

NUM_DETECTION_CLASSES = 49

PARTS_CLUSTERS = [
[1], # backside
[2], # beak
[3], # bliplate
[4], # body
[5, 18], # bwheel fwheel
[6], # cap
[7], # cbackside
[8], # chainwheel
[9], # cleftrightside
[10], # coach
[11], # croofside
[12], # door
[13], # engine
[14], # fliplate
[15], # frame
[16], # framescreen
[17], # frontside
[19], # hair
[20], # handlebar
[21], # head
[22], # headlight
[23], # hfrontside
[24], # hleftrightside
[25], # hroofside
[47], # mouth
[48], # muzzle
[49], # neck
[50], # nose
[51], # plant
[52], # pot
[26, 53], # lbhopa rbhopa
[27, 54], # lbleg rbleg
[28, 55], # lblleg rblleg
[29, 56], # lbfuleg rbuleg
[30, 57], # lear rear
[31, 58], # lebrow rebrow
[32, 67], # leftmirror rightmirror
[33, 68], # leftside rightside
[34, 59], # leye reye
[35, 60], # lfhopa rfhopa
[36, 61], # lfleg rfleg
[37, 62], # lflleg rflleg
[38, 63], # lfoot rfoot
[39, 65], # lhand rhand
[40, 66], # lhorn rhorn
[41, 69], # llarm rlarm
[42, 70], # lleg rleg
[43, 71], # llleg rlleg
[72], # roofside
[44, 73], # luarm ruarm
[45, 74], # luleg ruleg
[46, 75], # lwing rwing
[64], # rfuleg
[76], # saddle
[77], # stern
[78], # tail
[79], # torso
[80], # wheel
[81], # window
[255], # silh
]

DET_CATEGORIES = np.array([
0, # nocattegory
2, # aeroplane
259, # mountain
260, # mouse
415, # track
324, # road
9, # bag
258, # motorbike
144, # fence
18, # bed
19, # bedclothes
22, # bench
23, # bicycle
397, # diningtable
25, # bird
284, # person
158, # floor
31, # boat
416, # train
33, # book
34, # bottle
420, # tree
454, # window
295, # plate
296, # platform
427, # tvmonitor
44, # building
45, # bus
46, # cabinet
349, # shelves
232, # light
308, # pottedplant
440, # wall
59, # car
189, # ground
65, # cat
354, # sidewalk
424, # truck
68, # ceiling
326, # rock
72, # chair 
458, # wood
162, # food
207, # horse
80, # cloth
355, # sign
85, # computer
347, # sheep
220, # keyboard
159, # flower
360, # sky
98, # cow
187, # grass
104, # cup
105, # curtain
366, # snow
445, # water
368, # sofa
113, # dog
115, # door
])


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size = 4, growth_rate = 32, drop_rate = 0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()
        n_feat = num_init_features

        # First convolution
        self.stride_2 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, n_feat, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(n_feat)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        self.stride_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stride_4.add_module('denseblock1', _DenseBlock(num_layers=block_config[0], num_input_features=n_feat))
        n_feat = n_feat + block_config[0] * growth_rate
        self.stride_4.add_module('transition1', _Transition(num_input_features=n_feat, num_output_features=n_feat // 2))
        n_feat = n_feat // 2

        self.stride_8 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.stride_8.add_module('denseblock2', _DenseBlock(num_layers=block_config[1], num_input_features=n_feat))
        n_feat = n_feat + block_config[1] * growth_rate
        self.stride_8.add_module('transition2', _Transition(num_input_features=n_feat, num_output_features=n_feat // 2))
        n_feat = n_feat // 2

        self.stride_16 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.stride_16.add_module('denseblock3', _DenseBlock(num_layers=block_config[2], num_input_features=n_feat))
        n_feat = n_feat + block_config[2] * growth_rate
        self.stride_16.add_module('transition3', _Transition(num_input_features=n_feat, num_output_features=n_feat // 2))
        n_feat = n_feat // 2

        self.stride_32 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.stride_32.add_module('denseblock4', _DenseBlock(num_layers=block_config[3], num_input_features=n_feat))
        n_feat = n_feat + block_config[3] * growth_rate
        self.stride_32.add_module('norm5', nn.BatchNorm2d(n_feat))
        self.stride_32.add_module('relu5', nn.ReLU(True))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Load model from the checkpoint
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
        own_state = self.state_dict()
        for name_saved, param in state_dict.items():
            found = False
            name_saved = name_saved[9:]
            for name in own_state:
                if name[9:] == name_saved or name[10:] == name_saved:
                    own_state[name].copy_(param)
                    found = True
                    continue
            if not found:
                print('Did not find feature to assign: ', name_saved)

    def forward(self, x):
        str2 = self.stride_2(x)
        str4 = self.stride_4(str2)
        str8 = self.stride_8(str4)
        str16 = self.stride_16(str8)
        str32 = self.stride_32(str16)
        return str32, str16, str8, str4, str2


class SSD512Densenet(nn.Module):

    def __init__(self):
        super(SSD512Densenet, self).__init__()
        self.num_of_boxes = 6

        self.extractor = DenseNet()

        self.c_tr_str16 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, stride=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.Upsample(size=(32, 32), mode='bilinear'))
        self.c_str16 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.c_tr_str8 = nn.Sequential(nn.ConvTranspose2d(512, 256, 3, stride=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.Upsample(size=(64, 64), mode='bilinear'))
        self.c_str8 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.c_tr_str4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Upsample(size=(128, 128), mode='bilinear'))
        self.c_str4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.c_tr_str2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Upsample(size=(256, 256), mode='bilinear'))
        self.c_str2 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))

        self.c_skip_str16 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.c_skip_str8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.c_skip_str4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, bias=False, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.c_skip_str2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        self.det_loc_layer = nn.Conv2d(128, self.num_of_boxes * 4, 3, padding=1)
        self.det_cls_layer = nn.Conv2d(128, self.num_of_boxes * NUM_DETECTION_CLASSES, 3, padding=1)
        self.seg_layer = nn.Conv2d(128, 61, 3, padding=1)
        self.parts_layer = nn.Conv2d(128, 61, 3, padding=1)
        self.bounds_layer = nn.Conv2d(128, 1, 3, padding=1)
        self.sem_bounds_layer = nn.Conv2d(128, 20, 3, padding=1)

    def forward(self, x, is_training = True, det_boxes_enc = None, det_labels_enc = None, seg_gt = None, parts_gt = None, bounds_gt = None, sem_bounds_gt = None):

        x, str16, str8, str4, str2 = self.extractor(x)
        x = self.c_str16(torch.cat((self.c_tr_str16(x), self.c_skip_str16(str16)), 1))
        x = self.c_str8(torch.cat((self.c_tr_str8(x), self.c_skip_str8(str8)), 1))
        x = self.c_str4(torch.cat((self.c_tr_str4(x), self.c_skip_str4(str4)), 1))
        x = self.c_str2(torch.cat((self.c_tr_str2(x), self.c_skip_str2(str2)), 1))
        x = torch.nn.Upsample(size=(512, 512), mode='bilinear')(x)

        det_loc = self.det_loc_layer(x).permute(0,2,3,1).contiguous()
        det_boxes_pred = det_loc.view(det_loc.size(0), -1, 4)
        det_cls = self.det_cls_layer(x).permute(0,2,3,1).contiguous()
        det_label_pred = det_cls.view(det_cls.size(0),-1, NUM_DETECTION_CLASSES)
        seg_pred = self.seg_layer(x)
        parts_pred = self.parts_layer(x)
        bounds_pred = self.bounds_layer(x)
        sem_bounds_pred = self.sem_bounds_layer(x)

        if not is_training:
            return det_boxes_pred, det_label_pred, seg_pred, parts_pred, bounds_pred, sem_bounds_pred

        # detection classification loss
        pos = det_labels_enc > 0  # [N, #anchors]
        cls_loss = F.cross_entropy(det_label_pred.view(-1, NUM_DETECTION_CLASSES), det_labels_enc.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = cls_loss.view(pos.size(0), -1)
        cls_loss[det_labels_enc<0] = 0  # set ignored loss to 0
        # hard negative mining
        cls_loss_m_one = cls_loss * (pos.float() - 1)
        _, idx = cls_loss_m_one.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]
        num_neg = 3 * pos.long().sum(1)  # [N,]
        neg = rank < num_neg[:,None]   # [N,#anchors]
        cls_loss = cls_loss[pos|neg].mean()

        # detection localization loss
        mask = pos.unsqueeze(2).expand_as(det_boxes_pred) # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(det_boxes_pred[mask], det_boxes_enc[mask])
            
        # seg parts bound semseg losses
        seg_loss = F.nll_loss(F.log_softmax(seg_pred, dim=1), seg_gt)
        parts_loss = F.nll_loss(F.log_softmax(parts_pred, dim=1), parts_gt)
        bounds_loss = F.binary_cross_entropy(F.sigmoid(bounds_pred.squeeze(1)), bounds_gt)
        sem_bounds_loss = F.binary_cross_entropy(F.sigmoid(sem_bounds_pred), sem_bounds_gt)

        return cls_loss, loc_loss, seg_loss, parts_loss, bounds_loss, sem_bounds_loss


def map_on_cluster(x, clusters):
    i = 0
    for elem in clusters:
        if x in elem:
            return i
        i = i + 1


def get_bboxes(data):
    ids = np.array([2,23,25,31,34,45,59,65,72,98,397,113,207,258,284,308,347,368,416,427,9,18,22,33,44,46,80,85,104,115,144,159,162,220,232,259,260,105,355,295,326,349,19,415,424,440,454,458])
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


def create_parts_dict(data):
    final_dict = {}
    im2elem = {}
    im2shape = {}
    for elem in data['annos_segmentation']:
        img = elem['image_id']
        if not img in im2elem.keys():
            im2elem[img] = []
        if not img in im2shape.keys():
            im2shape[img] = elem['segmentation']['size']
        im2elem[img].append(elem['parts'])
    i = 0
    for name in im2elem.keys():
        if i%100==0:
            print(i)
        i = i+1
        temp = im2elem[name]
        all_img = np.zeros(im2shape[name], dtype=np.uint8)
        for parts in temp:
            for part in parts:
                cat = map_on_cluster(part['part_id'], [[0]] + PARTS_CLUSTERS)
                m = mask.decode(part['segmentation'])
                all_img[np.nonzero(m)] = cat
        final_dict[name] = all_img
    return final_dict


class VOCClassPascal(data.Dataset):

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    class_names = np.array([
        'nocat',
        'aeroplane',
        'mountain',
        'mouse',
        'track',
        'road',
        'bag',
        'motorbike',
        'fence',
        'bed',
        'bedclothes',
        'bench',
        'bicycle',
        'diningtable',
        'bird',
        'person',
        'floor',
        'boat',
        'train',
        'book',
        'bottle',
        'tree',
        'window',
        'plate',
        'platform',
        'tvmonitor',
        'building',
        'bus',
        'cabinet',
        'shelves',
        'light',
        'pottedplant',
        'wall',
        'car',
        'ground',
        'cat',
        'sidewalk',
        'truck',
        'ceiling',
        'rock',
        'chair',
        'wood',
        'food',
        'horse',
        'cloth',
        'sign',
        'computer',
        'sheep',
        'keyboard',
        'flower',
        'sky',
        'cow',
        'grass',
        'cup',
        'curtain',
        'snow',
        'water',
        'sofa',
        'dog',
        'door',
    ])


    def __init__(self, root, box_coder, split='train', transform=False, load_cache = True):
        self.root = root
        self.box_coder = box_coder
        self.split = split+"_list"
        self._transform = transform
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2010')
        self.details = Detail(dataset_dir + "/trainval_withkeypoints.json", "", "trainval")
        # VOC2011 and others are subset of VOC2012
        self.files = collections.defaultdict(list)
        self.detections, self.labels = get_bboxes(self.details.data)
        self.change_labels = np.vectorize(lambda x: np.where(DET_CATEGORIES == x)[0][0])
        if not load_cache:
            self.parts_dict = create_parts_dict(self.details.data)
            pickle.dump(self.parts_dict, open(str(split) + "parts.pickle", "wb"), protocol=2)
        else:
            self.parts_dict = pickle.load(open(str(split) + "parts.pickle", "rb"))
        self.sem_bound = json.load(open(dataset_dir+"/result.json"))
        for split in ['train_list', 'val_list']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                did = str(did)[:4]+"_"+str(did)[4:]
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                self.files[split].append({
                    'id': str(did),
                    'img': img_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        shape = img.shape
        # load label
        img_name = data_file['id']
        seg_gt = self.change_labels(self.details.getMask(img_name, show=False)).astype(np.int32)
        img_id = int(img_name[:4] + img_name[5:])
        detection = self.detections[img_id]
        label = self.labels[img_id]

        parts = self.parts_dict[img_id]
        bounds = self.details.getBounds(img_name, show=False)
        bounds = np.array(bounds)[np.newaxis,:,:]
        sb = self.sem_bound[img_name]
        sem_bounds = np.array([mask.decode(sb[i]) for i in sb])
        if self._transform:
            return self.transform(img, detection, label, img_id, seg_gt, shape, parts, bounds, sem_bounds)
        else:
            return img, detection, label, img_id, seg_gt,shape, parts, bounds, sem_bounds

    def transform(self, img, lbl, lbl2, im_id, seg_gt, shape, parts, bounds, sem_bounds):
        w,h,_ = img.shape
        new_img = np.zeros((512,512,3), dtype=np.uint8)
        new_img[:w,:h,:] = img
        img = new_img

        seg_gt_transformed = np.zeros((512,512), dtype=np.int32)
        seg_gt_transformed[:w,:h] = seg_gt

        parts_gt_transformed = np.zeros((512,512), dtype=np.int32)
        parts_gt_transformed[:w,:h] = parts

        bounds_gt_transformed = np.zeros((512,512), dtype=np.int32)
        bounds_gt_transformed[:w,:h] = bounds

        sem_bounds_gt_transformed = np.zeros((20,512,512), dtype=np.int32)
        sem_bounds_gt_transformed[:,:w,:h] = sem_bounds

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        lbl2 = torch.from_numpy(lbl2).long()
        seg_gt_transformed = torch.from_numpy(seg_gt_transformed).long()
        parts_gt_transformed = torch.from_numpy(parts_gt_transformed).long()
        bounds_gt_transformed = torch.from_numpy(bounds_gt_transformed).float()
        sem_bounds_gt_transformed = torch.from_numpy(sem_bounds_gt_transformed).float()

        lbl, lbl2 = self.box_coder.encode(lbl, lbl2)

        return img, lbl, lbl2, im_id, seg_gt_transformed, shape, parts_gt_transformed, bounds_gt_transformed, sem_bounds_gt_transformed

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


class SSDBoxCoder:
    def __init__(self):
        self.steps = (1,)
        self.box_sizes = (96.0, 256.0)
        self.aspect_ratios = ((2,),)
        self.fm_sizes =  (512,)
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                for s in self.box_sizes:
                    boxes.append((cx, cy, s, s))
                    for ar in self.aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.
        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''

        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        if len(boxes):
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
        else:
            boxes = torch.FloatTensor([])
            labels = torch.IntTensor([])
            scores = torch.FloatTensor([])

        return boxes, labels, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--load_cache')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    assert torch.cuda.is_available()
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    model = torch.nn.DataParallel(SSD512Densenet()).cuda()
    optim = torch.optim.SGD(model.parameters(), lr=0.0075, momentum=0.9, weight_decay=0.0001)
    box_coder = SSDBoxCoder()

    if args.epochs > 0:
        train_loader = torch.utils.data.DataLoader(
            VOCClassPascal('data/datasets', box_coder, split='train', transform=True, load_cache = args.load_cache),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # training
    for epoch in range(0, args.epochs):
        for batch_idx, (data, det_boxes_enc, det_labels_enc, _, seg_gt, _, parts_gt, bounds_gt, sem_bounds_gt) in enumerate(train_loader):

            assert model.training
            optim.zero_grad()

            data = Variable(data.cuda())
            det_boxes_enc = Variable(det_boxes_enc.cuda())
            det_labels_enc = Variable(det_labels_enc.cuda())
            seg_gt = Variable(seg_gt.cuda())
            parts_gt = Variable(parts_gt.cuda())
            bounds_gt = Variable(bounds_gt.cuda())
            sem_bounds_gt = Variable(sem_bounds_gt.cuda())

            cls_loss, loc_loss, seg_loss, parts_loss, bounds_loss, sem_bounds_loss = model(data, True, det_boxes_enc, det_labels_enc, seg_gt, parts_gt, bounds_gt, sem_bounds_gt)
            loss = cls_loss + loc_loss + seg_loss + parts_loss + bounds_loss + sem_bounds_loss
            loss = loss.sum() / len(data)
            loss.backward()
            optim.step()

            if batch_idx % 10 == 0:
                print('\n\n')
                print('epoch', epoch)
                print('iteration in the epoch', batch_idx)
                print('loss', loss.data[0])
                print('cls_loss', cls_loss.data[0])
                print('loc_loss', loc_loss.data[0])
                print('seg_loss', seg_loss.data[0])
                print('parts_loss', parts_loss.data[0])
                print('bounds_loss', bounds_loss.data[0])
                print('sem_bounds_loss', sem_bounds_loss.data[0])

    torch.save(model, 'multitask.pth')
    #model = torch.load('multitask_backup.pth')

    # validation dataset
    val_loader = torch.utils.data.DataLoader(
        VOCClassPascal('data/datasets', box_coder, split='val', transform=True, load_cache = args.load_cache),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    # validation dataset prediction
    if True:
        model.eval()
        results = {}
        for batch_idx, (data, _, _, im_id, _, img_shape, _, _, _) in enumerate(val_loader):
            print('batch index: ', batch_idx)
            sys.stdout.flush()
            det_boxes_pred, det_label_pred, seg_pred, parts_pred, bounds_pred, sem_bounds_pred = model(Variable(data.cuda()), False)

            # detection prediction
            det_boxes_pred = det_boxes_pred.data.cpu()
            det_labels_pred = F.softmax(det_label_pred, dim=2).data.cpu()
            det_boxes_pred, det_labels_pred, det_probs_pred = box_coder.decode(det_boxes_pred[0], det_labels_pred[0])
            im_pred = {}
            im_pred["bbox"] = det_boxes_pred.numpy().tolist()
            im_pred["labels"] = det_labels_pred.numpy().tolist()
            im_pred["probs"] = det_probs_pred.numpy().tolist()

            # segmentation prediction
            seg_pred = seg_pred.data.max(1)[1].cpu().numpy()[:, :, :]
            seg_pred = seg_pred[:,:int(img_shape[0]), :int(img_shape[1])]
            im_pred['segm'] = []
            for i in range(len(DET_CATEGORIES)):
                islice = 1*(seg_pred==i)
                if np.sum(islice.flatten())>0:
                    new_cat = {}
                    new_cat['id'] = DET_CATEGORIES[i]
                    new_cat['mask'] = mask.encode(np.asfortranarray(np.array(islice[0], dtype=np.uint8)))
                    new_cat['mask']['counts'] = new_cat['mask']['counts'].decode("utf-8")
                    im_pred['segm'].append(new_cat)

            # parts prediction
            parts_pred = parts_pred.data.max(1)[1].cpu().numpy()[:, :, :]
            parts_pred = parts_pred[:,:int(img_shape[0]), :int(img_shape[1])]
            im_pred['parts'] = []
            for i in range(len(PARTS_CLUSTERS)):
                islice = 1*(parts_pred==i)
                if np.sum(islice.flatten())>0:
                    new_part = {}
                    new_part['part_id'] = int(np.min(PARTS_CLUSTERS[i]))
                    new_part['mask'] = mask.encode(np.asfortranarray(np.array(islice[0], dtype=np.uint8)))
                    new_part['mask']['counts'] = new_part['mask']['counts'].decode("utf-8")
                    im_pred['parts'].append(new_part)

            # bounds prediction
            bound_thresh = 1 * (bounds_pred.data.cpu() > 0)
            im_pred['bound'] = mask.encode(np.asfortranarray(np.array(bound_thresh[0][0], dtype=np.uint8)))
            im_pred['bound']['counts'] = im_pred['bound']['counts'].decode("utf-8")

            # sem bounds prediction
            sem_bound_thresh = 1 * (sem_bounds_pred.data.cpu() > 0)
            im_pred['sem_bound'] = []
            for i in range(20):
                res = mask.encode(np.asfortranarray(np.array(sem_bound_thresh[0][i], dtype=np.uint8)))
                res['counts'] = res['counts'].decode("utf-8")
                im_pred['sem_bound'].append(res)

            results[str(im_id.numpy()[0])] = im_pred

        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        name = 'TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
        with open(name + '.json', 'a') as f:
            multitask_utils.print_dict_types(results)
            json.dump(results, f)

if __name__ == '__main__':
    main()
