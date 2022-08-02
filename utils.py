import json
import numpy as np
from tqdm import tqdm
from sklearn import metrics

def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split()
            else:
                lines=line.strip().split()
            yield lines

def readRecData(path,test_ratio=0.2):
    print('Reading DDI triplets...')
    drug1_set,drug2_set=set(),set()
    DDI=[]
    for d1,  d2, label in tqdm(readTriple(path)):
        drug1_set.add(int(d1))
        drug2_set.add(int(d2))
        DDI.append((int(d1),int(d2),int(label)))
    return list(drug1_set),list(drug2_set),DDI

def write_log(filename: str, log, mode='w'):
    with open(filename, mode) as writers:
        writers.write('\n')
        json.dump(log, writers, indent=4, ensure_ascii=False)


def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)  #roc曲线下面积
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision) #pr曲线下面积
    return pr_auc

def scores(y, pred):
    f1 = metrics.f1_score(y_true=y, y_pred=pred)
    acc = metrics.accuracy_score(y_true=y, y_pred=pred)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(y)):
        if y[j] == 1:
            if y[j] == pred[j]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y[j] == pred[j]:
                tn = tn + 1
            else:
                fp = fp + 1
    if tp == 0 and fp == 0:
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        precision = 0
        MCC = 0
    else:
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        precision = float(tp) / (tp + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))

    Pre = np.float64(precision)
    Sen = np.float64(sensitivity)
    Spe = np.float64(specificity)
    MCC = np.float64(MCC)
    f1 = np.float64(f1)
    acc = np.float64(acc)

    return f1, acc, Pre, Sen, Spe, MCC
