# coding:utf-8

import numpy as np
import math
from scipy import sparse
from sklearn import svm
from skmultilearn import problem_transform as pt
from sklearn import metrics

eps = 1E-7

class LabCor:
    
    def __init__(self, max_iter=1E7, meta_mode=2, C=1, map_size=100, imb_rate = 0.05, neg_weight=[0.5, 0.2], mask_rate=2, decay_rate=10, slope=0.5):
        
        self.max_iter = max_iter
        self.meta_mode = meta_mode
        self.C = C
        self.imb_rate = imb_rate
        self.neg_weight_balanced = neg_weight[0]
        self.neg_weight_imbalanced = neg_weight[1]
        self.mask_rate = mask_rate
        self.map_size = map_size
        self.decay_rate = decay_rate
        self.slope=slope
        
    def fit(self, x_tr, y_tr):
        
        y_total_dim = np.size(y_tr, axis=1)
        non_zero_dim = np.sum(y_tr, axis=0) != 0
        y_tr = np.array(y_tr)[:, non_zero_dim]
        y_dim = np.size(y_tr, axis=1)           # y空间的维度
        m_tr = np.size(y_tr, axis=0)            # 训练集的数量
        
        # params
        max_iter = self.max_iter
        map_size = self.map_size
        meta_mode = self.meta_mode        
        imb_rate = self.imb_rate
        neg_weight_balanced = self.neg_weight_balanced
        neg_weight_imbalanced = self.neg_weight_imbalanced
        mask_rate = self.mask_rate
        decay_rate = self.decay_rate
        slope = self.slope
        
        # slope along x and y direction
        slope_x = slope / (slope ** 2 + 1) ** (1/2)
        slope_y = 1 / (slope ** 2 + 1) ** (1/2)
        # ------------------- training ------------------
        
        # train base classifiers
        br_clf = pt.BinaryRelevance(svm.LinearSVC(max_iter=max_iter, C=self.C))
        x_tr = np.array(x_tr)
        y_tr = np.int32(y_tr)
        br_clf.fit(x_tr, y_tr)
        base_clfs = br_clf.classifiers_
        
        # train meta classifiers
        meta_clfs = [None for _ in range(y_dim)]
        for dim in range(y_dim):
            # expand dimensions
            x = self.generate_meta_x(x_tr, y_tr, dim, mode = meta_mode)
            y = y_tr[:, dim]
            clf = svm.LinearSVC(max_iter=max_iter, C=self.C)
            clf.fit(x, y)
            meta_clfs[dim] = clf
    
        # --------------- get predictions ----------------
        # get base predictions
        base_proba = np.vstack([self.sigmoid(clf.decision_function(x_tr)) for clf in base_clfs]).T
        base_pred = np.where(base_proba > 0.5, 1, 0)
        
        # positions for meta predictions
        meta_proba = np.zeros(base_proba.shape)
        meta_pred = np.zeros(base_pred.shape)
        
        # get meta predictions
        for dim in range(y_dim):   
            
            clf = meta_clfs[dim]
            x = self.generate_meta_x(x_tr, base_pred, dim, mode = meta_mode)
            proba = self.sigmoid(clf.decision_function(x))
            meta_proba[:, dim] = proba
        
        # 真值
        true_inds = y_tr > 0.5
    
        # ------------- inner validation ----------------
        
        true_count = np.sum(true_inds, axis=1)
        lr_min = min(true_count)
        
        map_background_list = []
        map_adapt_list = []
        
        map_list = []
        scale_list = []
        position_list = []
        gradient_list = []
        
        for dim in range(y_dim):
            
            # 坐标变换
            draw_x = meta_proba[:, dim] - 0.5
            draw_y = base_proba[:, dim] - meta_proba[:, dim]
            
            x_max = max(draw_x)
            x_min = min(draw_x)
            x_width = x_max - x_min
            y_max = max(draw_y)
            y_min = min(draw_y)
            y_width = y_max - y_min
            
            # calculate a value to balance the scale between x axis and y axis
            scale = max((np.mean(draw_x ** 2) / (np.mean(draw_y ** 2) + eps)) ** (1/2), 1)
            scale_list.append(scale)
            
            # get the size of the of the map
            x_size = map_size
            y_size = int(x_size / x_width * y_width * scale)
            
            # get the radius of the Gauss mask
            gauss_mask_r = np.mean(np.abs(draw_x)) * mask_rate
            gauss_mask_R = int(gauss_mask_r / x_width * map_size)
            
            # create Gauss mask
            mask_x = mask_y = np.arange(- gauss_mask_R, gauss_mask_R + 1)
            mask_X, mask_Y = np.meshgrid(mask_x, mask_y)
            dist = np.sqrt(mask_X ** 2 + mask_Y ** 2)
            gauss_mask = np.where(dist > gauss_mask_R, 0, np.exp(- decay_rate * dist / gauss_mask_R))
            
            # gradient
            gradient = gauss_mask[int(4 * gauss_mask_R / 5) + 1, gauss_mask_R] - gauss_mask[int(4 * gauss_mask_R / 5), gauss_mask_R]
            gradient_x = gradient * slope_x
            gradient_y = gradient * slope_y / scale
            x_center = int((- x_min / x_width) * x_size) + gauss_mask_R
            y_center = int((- y_min / y_width) * y_size) + gauss_mask_R
            gradient_list.append([gradient_x, gradient_y])
            
            # draw background confidence map
            gauss_X, gauss_Y = np.meshgrid(range(x_size + gauss_mask_R * 2), range(y_size + gauss_mask_R * 2))
            map_background = ((gauss_X - x_center) * gradient_x + (y_center - gauss_Y) * gradient_y).T 
            map_background_list.append(map_background.copy())
            
            # automatically select a weight for the current dimension based on label imbalance condition
            neg_weight = neg_weight_imbalanced if np.sum(true_inds[:, dim]) / m_tr < imb_rate else neg_weight_balanced
            
            # draw self-adapted confidence map
            map_adapt = np.zeros(map_background.shape)
            for i, _ in enumerate(draw_x):
                    
                    # change from probalistic value into a coordinate on the map
                    x_position = int(math.floor((draw_x[i] - x_min) / (x_width + eps) * x_size)) + gauss_mask_R
                    y_position = int(math.floor((draw_y[i] - y_min) / (y_width + eps) * y_size)) + gauss_mask_R
                    
                    # add a positive confidence to the surrounding area of a positive sample
                    if true_inds[i, dim]:
                        map_adapt[x_position-gauss_mask_R:x_position+gauss_mask_R + 1, 
                                  y_position-gauss_mask_R:y_position+gauss_mask_R + 1] += gauss_mask * meta_proba[i, dim]
                    # add a negative confidence to the surrounding area of a negative sample
                    else:
                        map_adapt[x_position-gauss_mask_R:x_position+gauss_mask_R + 1, 
                                  y_position-gauss_mask_R:y_position+gauss_mask_R + 1] -= gauss_mask * neg_weight * (1-meta_proba[i, dim])                     
            map_adapt_list.append(map_adapt.copy())
            
            # the final decision pattern
            map_list.append(map_adapt + map_background)
            
            # store the key values for the affine transformation
            position_list.append(np.array([x_min, x_width, y_min, y_width, gauss_mask_r, gauss_mask_R, x_size, y_size]))
            
        self.non_zero_dim = non_zero_dim
        self.y_dim = y_dim
        self.y_total_dim = y_total_dim
        self.br_clf = br_clf
        self.base_clfs = base_clfs
        self.meta_clfs = meta_clfs
        self.map_list = map_list
        self.map_background_list = map_background_list
        self.map_adapt_list = map_adapt_list
        self.scale_list = scale_list
        self.position_list = position_list
        self.lr_min = lr_min
        self.gradient_list = gradient_list
        
    def predict(self, x_te):
        
        m = np.size(x_te, axis=0)
        map_list = self.map_list
        scale_list = self.scale_list
        position_list = self.position_list
        gradient_list = self.gradient_list
        lr_min = self.lr_min
        y_dim = self.y_dim
        
        # ------------------- get base predictions --------------------
        base_proba = np.vstack([self.sigmoid(clf.decision_function(x_te)) for clf in self.base_clfs]).T
        base_pred = np.where(base_proba > 0.5, 1, 0)
        
        # ------------------- get meta predictions --------------------
        meta_proba = np.zeros(base_proba.shape)
        meta_pred = np.zeros(base_pred.shape)
        
        for dim in range(self.y_dim):

            clf = self.meta_clfs[dim]
            x = self.generate_meta_x(x_te, base_pred, dim, mode = self.meta_mode)
            proba = self.sigmoid(clf.decision_function(x))
            meta_proba[:, dim] = proba
        
        meta_pred = np.where(meta_proba > 0.5, 1, 0)
        
        # ------------------ label correction ----------------------------
        
        draw_x = meta_proba - 0.5
        draw_y = base_proba - meta_proba
        
        confidence = np.zeros(base_proba.shape)
        
        for dim in range(y_dim):
            
            gauss_map = map_list[dim]
            scale = scale_list[dim]
            [gradient_x, gradient_y] = gradient_list[dim]
            [x_min, x_width, y_min, y_width, gauss_mask_r, gauss_mask_R, x_size, y_size] = position_list[dim]

            for i, _ in enumerate(draw_x):
                
                # get position for each sample
                x_position = int(math.floor((draw_x[i, dim] - x_min) / (x_width + eps) * x_size) + gauss_mask_R)
                y_position = int(math.floor((draw_y[i, dim] - y_min) / (y_width + eps) * y_size) + gauss_mask_R)
                
                # if the sample is in the region of interest
                if x_position < gauss_map.shape[0] and x_position >= 0 and y_position < gauss_map.shape[1] and y_position >= 0: 
                    confidence[i, dim] = gauss_map[x_position, y_position]
                # if the sample is not in the region of interest
                else:
                    confidence[i, dim] = draw_x[i, dim] * gradient_x * x_size / x_width - draw_y[i, dim] * gradient_y * y_size / y_width
                    
        predictions = confidence > 0

        # correction based on label count constraint
        for i, _ in enumerate(predictions):
             
            if lr_min > np.sum(predictions[i]):
                
                for _ in range(lr_min):
                    
                    top_ind = np.argmax(confidence[i])
                    predictions[i, top_ind] = True
                    confidence[i, top_ind] = -1E5
        
        output = np.zeros([m, self.y_total_dim])
        output[:, self.non_zero_dim] = predictions
        
        return output

    
    @staticmethod
    def generate_meta_x(x, y, dim, mode = 2):

        # 将第dim维的标签作为输出，其他标签作为输入
        dim_rest = list(range(y.shape[1]))
        dim_rest.remove(dim)

        if mode == 1:
            return np.hstack([x, np.where(y[:, dim_rest] == 1, 1, -1)])
        if mode == 2:
            return np.hstack([x, y[:, dim_rest]])
        if mode == 3:
            return np.where(y[:, dim_rest] == 1, 1, -1)
        if mode == 4:
            return y[:, dim_rest]
    
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0+1.0/np.exp(x))

class Evaluate:

    def __init__(self):

        return

    @staticmethod
    def eval_hamming_loss(y, y_):

        return metrics.hamming_loss(y, y_)

    @staticmethod
    def eval_accuracy(y, y_):
        
        return metrics.jaccard_score(y, y_, average = 'samples')

    @staticmethod
    def eval_exact_match(y, y_):
        
        dim = np.size(y, axis=1)
        cor = np.sum(list(map(lambda a, b: a == b, y, y_)), axis=1)
        return sum(dim == cor) / len(y)

    @staticmethod
    def eval_f1(y, y_):
        
        return metrics.f1_score(y, y_, average = 'samples')
    
    @staticmethod
    def eval_macro_f1(y, y_):
        
        return metrics.f1_score(y, y_, average = 'macro')
    
    @staticmethod
    def eval_micro_f1(y, y_):

        return metrics.f1_score(y, y_, average = 'micro')
    
    def evaluator(self, y, y_, names):

        return {
            'hamming_loss': self.eval_hamming_loss(y, y_) if 'hamming_loss' in names else 0,
            'accuracy': self.eval_accuracy(y, y_) if 'accuracy' in names else 0,
            'exact_match': self.eval_exact_match(y, y_) if 'exact_match' in names else 0,
            'f1': self.eval_f1(y, y_) if 'f1' in names else 0,
            'macro_f1': self.eval_macro_f1(y, y_) if 'macro_f1' in names else 0,
            'micro_f1': self.eval_micro_f1(y, y_) if 'micro_f1' in names else 0,
        }
