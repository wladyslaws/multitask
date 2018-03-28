import numpy as np
import json
from detail import mask

class baseSegEvalClass(object):
    def __init__(self, details, verbose = True):
        self.details = details
        self.gt_dict = self._json_to_dict(self.details.data)
        self.data_loaded = False
        self.verbose = verbose

    #requires path to file with predictions
    def loadJSON(self, resFile):
        self.data = json.load(open(resFile))
        self.pred_dict = self._json_to_dict(self.data)
        self.cats = [2,259,260,415,324,9,258,144,18,19,22,23,397,25,284,158,31,416,33,34,420,454,295,296,427,44,45,46,349,232,308,440,59,189,65,354,424,68,326,72,458,162,207,80,355,85,347,220,159,360,98,187,104,105,366,445,368,113,115]
        self.data_loaded = True
        
        
    # Gets list of categories existing in prediction file
    def _get_cats(self, prediction):
        cats = set()
        for elem in prediction.values():
            cats = set.union(cats, set(elem.keys()))
        return list(cats)

    # Takes a JSON file and transforms it into dictionary which makes it easy to search for maps 
    # on certain image for certain class.
    def _json_to_dict(self, data):
        raise NotImplementedError()
    

    # Computes intersection and union given dicts returned by _json_to_dict
    def _IU(self, img1, img2, cat):
        hits1 = 1*(img1==cat)
        hits2 = 1*(img2==cat)
        intersection = int((hits1*hits2).sum())
        union = len(np.where((hits1 + hits2)>0)[0])
        return (intersection, union)
    
    # Performs computation of IoU between ground truth available in details and prediction
    # in file previously loaded with loadJSON. Writes the result under self.results
    def evaluate(self):
        if not self.data_loaded:
            print("Please, load the data with loadJSON first")
            return -1
        cats = self.cats
        dict_pred = self.pred_dict
        dict_gt = self.gt_dict
        intersection = np.zeros(len(cats))
        union = np.zeros(len(cats))
        j = 0
        if self.verbose:
            from progressbar import ProgressBar
            pbar = ProgressBar(maxval=len(dict_pred.keys())).start()
        for img in self.im_names:
            img1 = dict_pred[img]
            img2 = dict_gt[img]
            for i in range(len(cats)):
                cat = cats[i]
                temp_int, temp_un = self._IU(img1, img2, cat)
                intersection[i] = intersection[i] + temp_int
                union[i] = union[i] + temp_un
            if self.verbose:
                pbar.update(j)
            j = j + 1
        pbar.finish()

        self.intersection = intersection
        self.union = union
        self.results = np.nanmean(intersection/union)   

    def load_img_names(self, _file):
        #self.im_names = [2008000002, 2008000003, 2010004229, 2008006662, 2008000007, 2008000008, 2008005257, 2008004619, 2010005903, 2010005904, 2010004369, 2010005907, 2008004758, 2010003351, 2008005218, 2008004634, 2010004371, 2008004636, 2008000026, 2008000027, 2008003881, 2008000019, 2008006665, 2008004408, 2008003779, 2008000009, 2008004551, 2008004552, 2010003405, 2010003541, 2008000016, 2008004313, 2008000015, 2010004193, 2008005090, 2008004325, 2008005094, 2010004071, 2008000023, 2008004459, 2008004975, 2010004208, 2008006641, 2010004211, 2010004374, 2010004345, 2010004475, 2008000028, 2008006655]
        self.im_names = [int(x) for x in open(_file).read().split("\n")[:-1]]

    def get_shape(self, img):
        for elem in self.details.data['images']:
            if elem['image_id']==img:
                return (elem['height'], elem['width'])
        
# Class for evaluation of segmentation. dict[image][category] gives a list of maps (@D arrays).
# Each map is a map of a single instance of the category on the image.
class catSegEvalClass(baseSegEvalClass):
    def load_img_names(self, _file):
        #self.im_names = [2008000002, 2008000003, 2010004229, 2008006662, 2008000007, 2008000008, 2008005257, 2008004619, 2010005903, 2010005904, 2010004369, 2010005907, 2008004758, 2010003351, 2008005218, 2008004634, 2010004371, 2008004636, 2008000026, 2008000027, 2008003881, 2008000019, 2008006665, 2008004408, 2008003779, 2008000009, 2008004551, 2008004552, 2010003405, 2010003541, 2008000016, 2008004313, 2008000015, 2010004193, 2008005090, 2008004325, 2008005094, 2010004071, 2008000023, 2008004459, 2008004975, 2010004208, 2008006641, 2010004211, 2010004374, 2010004345, 2010004475, 2008000028, 2008006655]
        self.im_names = [int(x) for x in open(_file).read().split("\n")[:-1]]

    def _json_to_dict(self, data):
        final_dict = {}
        im2elem = {}
        for elem in data['annos_segmentation']:
            img = elem['image_id']
            if not img in im2elem.keys():
                im2elem[img] = []
            im2elem[img].append((elem['category_id'],elem['segmentation']))
        i = 0
        for name in im2elem.keys():
            if i%100==0:
                print(i)
            i = i+1
            temp = im2elem[name]
            all_img = np.zeros(temp[0][1]['size'], dtype=np.uint8)
            for cat, to_decode in temp:
                m = mask.decode(to_decode)
                all_img[np.nonzero(m)] = cat
            final_dict[name] = all_img
        
        return final_dict


# Each map is a map of a single instance of the category or part on the image.
# Here category is available in the dict only if it does not contains of any parts.
class partsSegEvalClass(baseSegEvalClass):
   
    def _map_on_cluster(self, x, clusters):
        for elem in clusters:
            if x in elem:
                return np.min(np.array(elem))
        return x
        
    def _json_to_dict(self, data):
        self.clusters = [[1], [2], [3], [4], [5, 18], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [19], [20], [21], [22], [23], [24], [25], [47], [48], [49], [50], [51], [52], [26,53], [27,54], [28, 55], [29,56], [30, 57], [31, 58], [32, 67], [33,68], [34, 59], [35, 60], [36, 61], [37, 62], [38,63], [39, 65], [40,66], [41, 69], [42, 70], [43, 71], [72], [44,73], [45, 74], [46, 75], [64], [76], [77], [78], [79], [80], [81], [255]]
        final_dict = {}
        im2elem = {}
        im2shape = {}
        i = 0
        for elem in data['annos_segmentation']:
            if i%100==0:
                print i
            i = i+1
            img = elem['image_id']
            if not img in im2elem.keys():
                im2elem[img] = []
            if not img in im2shape.keys():
                im2shape[img] = self.get_shape(img)
            im2elem[img].append(elem['parts'])
        i = 0
        for name in im2elem.keys():
            if i%100==0:
                print i
            i = i+1
            temp = im2elem[name]
            all_img = np.zeros(im2shape[name], dtype=np.uint8)
            for parts in temp:
                for part in parts:
                    cat = self._map_on_cluster(part['part_id'], self.clusters)
                    m = mask.decode(part['segmentation'])
                    all_img[np.nonzero(m)] = cat
            final_dict[name] = all_img
        return final_dict
