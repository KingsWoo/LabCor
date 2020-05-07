# coding:utf-8

import numpy as np
import math
from scipy import sparse
from sklearn import svm
from skmultilearn import problem_transform as pt
from sklearn import metrics
from sklearn.svm import SVC

class LabCor:
    
    def __init__(self, max_iter=5000, meta_mode=2):
        
        self.max_iter = max_iter
        self.meta_mode = meta_mode
    
    def fit(self, x_tr, y_tr):
        
        y_total_dim = np.size(y_tr, axis=1)
        non_zero_dim = np.sum(y_tr, axis=0) != 0
        y_tr = np.array(y_tr)[:, non_zero_dim]
        
        
        # params
        max_iter = self.max_iter
        meta_mode = self.meta_mode
        
        # base classifier
        pred = pt.BinaryRelevance(svm.LinearSVC(max_iter=max_iter))
        x_tr = np.array(x_tr)
        y_tr = np.int32(y_tr)
        pred.fit(x_tr, y_tr)
        y_tr_ = sparse.dok_matrix.toarray(pred.predict(x_tr)) # 训练集根据训练结果获得的预测值
        
        # meta
        y_dim = np.size(y_tr, axis=1)           # y空间的维度
        m_tr = np.size(y_tr, axis=0)            # 训练集的数量
        m_te = round(m_tr / 9)                  # 假想中测试集数量
        m_va = 3 * m_te                         # 验证集（确定优化顺序）数量
        m_vc = m_te                             # 验证集（确定截止位置）数量

        # 分割测试集与验证集
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)
        x_vc = np.array(x_tr)[m_va:m_va+m_vc, :]
        y_vc = np.array(y_tr)[m_va:m_va+m_vc, :]
        y_vc_ = np.array(y_tr_)[m_va:m_va + m_vc, :]
        x_va = np.array(x_tr)[0:m_va, :] 
        y_va = np.array(y_tr)[0:m_va, :]
        y_va_ = np.array(y_tr_)[0:m_va, :]
        
        # 初始化评估参数
        ev = Evaluate()
        best_pos_thres = np.ones([y_dim])
        best_neg_thres = np.zeros([y_dim])
        best_pos_impro = - np.ones([y_dim])
        best_neg_impro = - np.ones([y_dim])
        best_impro = np.zeros([y_dim])
        
        thres_pos_list = np.array(range(50, 101)) / 100
        thres_neg_list = np.array(range(50, -1, -1)) / 100

        classifiers = [None for _ in range(y_dim)]

        # ------------------ 获取每一维度标签矫正的效果 -------------------------------

        for dim in range(y_dim):

            # 将第dim维的标签作为输出，其他标签作为输入
            dim_rest = list(range(y_dim))
            dim_rest.remove(dim)

            # 重组训练集金标准的标签，获得训练结果
            x = self.generate_meta_x(x_tr, y_tr, dim, mode = meta_mode)
            y = y_tr[:, dim]
            if sum(y) == 0 or sum(y) == m_tr:
                continue
            classifier = SVC(probability=True, kernel='linear', max_iter=max_iter)
            classifier.fit(x, y)
            classifiers[dim] = classifier

            # 重组验证集的预测标签（y_va_），根据dim维度以外的标签获取该维度下该标签为1的概率
            x = self.generate_meta_x(x_va, y_va_, dim, mode = meta_mode)
            proba = classifier.predict_proba(x)[:, 1] if classifier is not None else 0.5 * np.ones([m_va])

            y_va_adv = y_va_.copy()

            # 扫描确定更改所使用的正阈值（0 -> 1）
            for thres_pos in thres_pos_list:

                y = [1 if proba[i] > thres_pos else y_va_[i, dim] for i in range(m_va)]
                imp = ev.improve_function(y_va[:, dim], y_va_[:, dim], y, names = ['single_label_f1', 'single_label_accuracy'])
                if imp > best_pos_impro[dim]:
                    best_pos_impro[dim] = imp
                    best_pos_thres[dim] = thres_pos

            # 扫描确定更改所使用的负阈值（1 -> 0）
            for thres_neg in thres_neg_list:

                y = [0 if proba[i] < thres_neg else y_va_[i, dim] for i in range(m_va)]
                imp = ev.improve_function(y_va[:, dim], y_va_[:, dim], y, names = ['single_label_f1', 'single_label_accuracy'])
                if imp > best_neg_impro[dim]:
                    best_neg_impro[dim] = imp
                    best_neg_thres[dim] = thres_neg

            # 取最佳的正负阈值联合确定针对该标签的矫正效果（有时只用到一个阈值）
            thres_pos = best_pos_thres[dim]
            thres_neg = best_neg_thres[dim]

            y = y_va_[:, dim].copy()

            y = [1 if proba[i] > thres_pos else y[i] for i in range(m_va)]
            y = [0 if proba[i] < thres_neg else y[i] for i in range(m_va)]

            best_impro[dim] = ev.improve_function(y_va[:, dim], y_va_[:, dim], y, names = ['single_label_f1', 'single_label_accuracy'])
        
        # ----------------------- 强标签列表 ----------------------------------
        
        strong_dim = np.arange(y_dim)[best_impro == 0]

        # ----------------------- 强标签首先纠正 ----------------------------------
        
        y_vc_adv = np.copy(y_vc_)

        for dim in strong_dim:

            thres_pos = best_pos_thres[dim]
            thres_neg = best_neg_thres[dim]
            classifier = classifiers[dim]

            # 矫正 y_vc_
            x = self.generate_meta_x(x_vc, y_vc_, dim, mode = meta_mode)
            proba = classifier.predict_proba(x)[:, 1] if classifier is not None else 0.5 * np.ones([m_vc])
            y = y_vc_adv[:, dim]
            y = [1 if proba[i] > thres_pos else y[i] for i in range(m_vc)]
            y = [0 if proba[i] < thres_neg else y[i] for i in range(m_vc)]
            y_vc_adv[:, dim] = y
            
        y_vc_ = y_vc_adv

        # ------------------- 根据改进效果确定纠正的顺序 ----------------------------
        
        # 提取具有改进效果的维度
        best_impro_cp = np.copy(best_impro)
        impro_dim = np.sum(np.where(best_impro > 0, True, False))

        # 确定改进顺序以及用于改进的维度
        change_sequence = np.zeros(impro_dim)
        for i in range(impro_dim):

            ind = np.argmax(best_impro_cp)
            change_sequence[i] = ind
            best_impro_cp[ind] = 0

        # ------------------------ 对y_vc_进行纠正 ---------------------------------

        # values_list用于记录各参考量的数据情况
        values_list = [None for _ in range(np.size(change_sequence) + 1)]

        y_vc_ori = np.copy(y_vc_)
        y_vc_adv = np.copy(y_vc_)

        criteria = ['hamming_loss', 'accuracy', 'exact_match', 'f1', 'macro_f1', 'micro_f1']
        values_list[0] = ev.improve_function(y_vc, y_vc_ori, y_vc_adv, names = criteria)

        # 逐次纠正适合纠正的各个标签维度
        for ind in range(np.size(change_sequence)):

            dim = int(change_sequence[ind])
            dim_rest = list(range(y_dim))
            dim_rest.remove(dim)

            classifier = classifiers[dim]
            x = self.generate_meta_x(x_vc, y_vc_adv, dim, mode = meta_mode)
            proba = classifier.predict_proba(x)[:, 1] if classifier is not None else 0.5 * np.ones([m_vc])
            thres_pos = best_pos_thres[dim]
            thres_neg = best_neg_thres[dim]
            y = y_vc_adv[:, dim]
            y = [1 if proba[i] > thres_pos else y[i] for i in range(m_vc)]
            y = [0 if proba[i] < thres_neg else y[i] for i in range(m_vc)]

            y_vc_adv[:, dim] = y

            values_list[ind+1] = ev.improve_function(y_vc, y_vc_ori, y_vc_adv, names = criteria)

        values_list[np.size(change_sequence)] = ev.improve_function(y_vc, y_vc_ori, y_vc_adv, names = criteria)
        # 确定截止位置
        cutoff_index = np.argmax(values_list)
        
        self.non_zero_dim = non_zero_dim
        self.base_clf = pred
        self.y_dim = y_dim
        self.y_total_dim = y_total_dim
        self.meta_clf = classifiers
        self.pos_thres = best_pos_thres
        self.neg_thres = best_neg_thres
        self.strong_dim = strong_dim
        self.change_sequence = change_sequence
        self.cutoff_index = cutoff_index
    
    def predict(self, x_te):
        
        m = np.size(x_te, axis=0)

        # ------------------------ base 获取 ---------------------------------
        y_base = sparse.dok_matrix.toarray(self.base_clf.predict(x_te))
        
        # ------------------------ 强标签矫正 ---------------------------------
        
        y_adv = np.copy(y_base)

        for dim in self.strong_dim:

            thres_pos = self.pos_thres[dim]
            thres_neg = self.neg_thres[dim]
            classifier = self.meta_clf[dim]

            # 矫正 y_vc_
            x = self.generate_meta_x(x_te, y_base, dim, mode = self.meta_mode)
            proba = classifier.predict_proba(x)[:, 1] if classifier is not None else 0.5 * np.ones([m])
            y = y_adv[:, dim]
            y = [1 if proba[i] > thres_pos else y[i] for i in range(m)]
            y = [0 if proba[i] < thres_neg else y[i] for i in range(m)]
            y_adv[:, dim] = y
            
        y_ = y_adv
        
        # ------------------------ 弱标签矫正 ---------------------------------
        
        y_adv = np.copy(y_)

        # 逐次纠正适合纠正的各个标签维度
        for ind in range(int(self.cutoff_index)):

            dim = int(self.change_sequence[ind])
            dim_rest = list(range(self.y_dim))
            dim_rest.remove(dim)

            classifier = self.meta_clf[dim]
            x = self.generate_meta_x(x_te, y_adv, dim, mode = self.meta_mode)
            proba = classifier.predict_proba(x)[:, 1] if classifier is not None else 0.5 * np.ones([m])
            thres_pos = self.pos_thres[dim]
            thres_neg = self.neg_thres[dim]
            y = y_adv[:, dim]
            y = [1 if proba[i] > thres_pos else y[i] for i in range(m)]
            y = [0 if proba[i] < thres_neg else y[i] for i in range(m)]

            y_adv[:, dim] = y     
        
        y_ = y_adv
        
        y_output = np.zeros([m, self.y_total_dim])
        y_output[:, self.non_zero_dim] = y_
        
        return y_output
        
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
    
    @staticmethod
    def eval_single_label_f1(y, y_):

        return metrics.f1_score(y, y_)
    
    @staticmethod
    def eval_single_label_accuracy(y, y_):
        
        return metrics.accuracy_score(y, y_)
    
    def evaluator(self, y, y_, names):

        return {
            'hamming_loss': self.eval_hamming_loss(y, y_) if 'hamming_loss' in names else 0,
            'accuracy': self.eval_accuracy(y, y_) if 'accuracy' in names else 0,
            'exact_match': self.eval_exact_match(y, y_) if 'exact_match' in names else 0,
            'f1': self.eval_f1(y, y_) if 'f1' in names else 0,
            'macro_f1': self.eval_macro_f1(y, y_) if 'macro_f1' in names else 0,
            'micro_f1': self.eval_micro_f1(y, y_) if 'micro_f1' in names else 0,
            'single_label_f1': self.eval_single_label_f1(y, y_) if 'single_label_f1' in names else 0,
            'single_label_accuracy': self.eval_single_label_accuracy(y, y_) if 'single_label_accuracy' in names else 0,
        }

    def improve_function(self, y, y_ori, y_adv, names):

        signs = [-1 if name == 'hamming_loss' else 1 for name in names]
        k = 1
        
        values_ori = self.evaluator(y, y_ori, names)
        values_adv = self.evaluator(y, y_adv, names)

        values_diff = [signs[i] * (values_adv[names[i]] - values_ori[names[i]]) for i in range(len(names))]
        values_base = [2 * math.sqrt(values_ori[names[i]] * (1 - values_ori[names[i]])) for i in range(len(names))]

        return sum([- math.exp(-k * values_diff[i]) + 1 for i in range(len(names))])

