import numpy as np
import readfile as rf
import math
import random
import labcor
import multiprocessing as mp
from sklearn import preprocessing as prep
import warnings

warnings.filterwarnings("ignore")

fold = 10
datasets = ['emotions']

ev = labcor.Evaluate()
criteria = ['hamming_loss', 'accuracy', 'exact_match', 'f1', 'macro_f1', 'micro_f1']

core_size = 50
random_seed = 20

max_iter = 10 ** 7

# define training function

def train(x, y, dataset, fold, k):

    # randomly index samples for k-fold cross validation
    
    m = int(math.floor(np.size(x, axis=0) / fold))
    random.seed(random_seed)
    indices = np.reshape(random.sample(range(m * fold), m * fold), [fold, m])
    
    group_tr = list(range(k+1, fold)) + list(range(k))
    group_te = k
        
    indice_tr = np.reshape(indices[group_tr], [m * (fold - 1)])
    indice_te = np.reshape(indices[group_te], [m])

    x_tr = x[indice_tr]
    y_tr = y[indice_tr]
    x_te = x[indice_te]
    y_te = y[indice_te]
        
    # define
    clf = labcor.LabCor(max_iter=max_iter)
        
    # training
    clf.fit(x_tr, y_tr)

    # prediction
    y_te_ = clf.predict(x_te)
    
    # evaluation
    eval_te = ev.evaluator(y_te_, y_te, criteria)

    # output results
    path = '../result/(dataset=%s) matrics.csv' % (dataset)
    eval_te = ev.evaluator(y_te_, y_te, criteria)
    rf.write_csv(path, [[k] + [eval_te[name] for name in criteria]], 'a')
    print('Finish: dataset %s fold %d' % (dataset, k))
    
# define process list
p_list = []

for dataset in datasets:

    # read dataset
    path_x = '../dataset/%s/%s_x.csv' % (dataset, dataset)
    path_y = '../dataset/%s/%s_y.csv' % (dataset, dataset)
    x, x_label = rf.read_csv(path_x)
    y, y_label = rf.read_csv(path_y)
    
    # pre-processing
    x = prep.scale(x, axis = 0)
    
    # fill process list
    for k in range(fold):
        p_list.append(mp.Process(target=train,args=(x, y, dataset, fold, k)))

# run multi-process with a defined core_size
for batch_number in range(math.ceil(len(p_list) / core_size)):
        
    for p in p_list[batch_number * core_size: (batch_number + 1) * core_size]:
        p.start()

    for p in p_list[batch_number * core_size: (batch_number + 1) * core_size]:
        p.join()            

