import numpy as np
import readfile as rf
import math
import random
import labcor as labcor
from sklearn import preprocessing as prep
import warnings

warnings.filterwarnings("ignore")

# ----------------- Setting Parameters -------------------

dataset = 'emotions'     # The name of the dataset
fold = 10                # total number of folds for cross validation
k = 0                    # the k-th cross validation
random_seed = 20         # random seed for cross validation

C = 1E-2                 # regulazation parameter for SVC classifier
max_iter = 1E7           # max iteration for SVC classifier
imb_rate = 0.05          # the positive rate for define a dimension is balanced / imbalanced
neg_weight = [0.5, 0.2]  # weight for negative samples in generating self-adapted confidence. 
                         # the first one is for balanced condition and the second one is for very imbalanced condition. 
mask_rate = 2            # control the size of mask. 2 is set for better competing.
map_size = 100           # the size of the matrix which stores the decision pattern along x-axis. the size along y-axis is calculated inside LabCor.
decay_rate = 10          # the decay rate of the Gauss mask
slope = 0.5              # the slope of the boundary in the background confidence map.

criteria = ['hamming_loss', 'accuracy', 'exact_match', 'f1', 'macro_f1', 'micro_f1'] # criteria used in evaluations
ev = labcor.Evaluate()   # an object for calculating the criteria

# ------------------- Reading Data ----------------------
# read dataset
path_x = '../dataset/%s/%s_x.csv' % (dataset, dataset)
path_y = '../dataset/%s/%s_y.csv' % (dataset, dataset)
x, x_label = rf.read_csv(path_x)
y, y_label = rf.read_csv(path_y)
    
# ------------------ pre-processing ---------------------
x = prep.scale(x, axis = 0)

# -------- split into training/testing sets -------------
m = int(math.floor(np.size(x, axis=0) / fold))
random.seed(random_seed)
indices = np.reshape(random.sample(range(m * fold), m * fold), [fold, m])

# 2x5 fold
group_tr = list(range(k, min(k+5, fold))) + list(range(0, k-5))              
group_te = list(range(k+5, min(k+10, fold))) + list(range(max(0, k-5), k))
    
indice_tr = np.reshape(indices[group_tr], [m * 5])
indice_te = np.reshape(indices[group_te], [m * 5])
    
x_tr = x[indice_tr]
y_tr = y[indice_tr]
x_te = x[indice_te]
y_te = y[indice_te]
        
# -------------------- training --------------------------
clf = labcor.LabCor(
    max_iter = max_iter, 
    C = C, 
    imb_rate = imb_rate,
    neg_weight = neg_weight,
    map_size = map_size, 
    mask_rate = mask_rate, 
    decay_rate = decay_rate, 
    slope = slope
)                        
  
# fitting
clf.fit(x_tr, y_tr)

# ------------------- prediction --------------------------
# prediction
y_te_ = clf.predict(x_te)

# ------------------- evaluation --------------------------
eval_te = ev.evaluator(y_te_, y_te, criteria)
print(eval_te)
