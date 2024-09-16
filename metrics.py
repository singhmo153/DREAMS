from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import pdb

def calculate_weighted_f1(y_hat, y):
    return f1_score(y, y_hat, average='weighted')

def calculate_weighted_recall(y_hat, y):
    return recall_score(y, y_hat, average='weighted')

def calculate_weighted_precision(y_hat, y):
    return precision_score(y, y_hat, average='weighted')

def calculate_accuracy(y_hat, y):
    score = accuracy_score(y_hat, y)
    return score

def report(y_hat, y):
    report = classification_report(y, y_hat)
    return report

def calculate_f1_binary(y_hat, y):
    score = f1_score(y, y_hat)
    return score

def calculate_f1_multiclass(y_hat, y):
    return f1_score(y, y_hat, average=None)

def calculate_class_wise_accuracy(y_hat, y):
    y_hat = np.array(y_hat)
    y = np.array(y)

    classwise_accuracy = {}

    unique_classes = np.unique(y)

    for class_label in unique_classes:
        class_mask = y == class_label
        class_predicted = y_hat[class_mask]
        class_true = y[class_mask]

        class_accuracy = accuracy_score(class_true, class_predicted)

        classwise_accuracy[class_label] = class_accuracy

    return classwise_accuracy

def binary_class_auc(y_hat, y):
    score_w = roc_auc_score(y, y_hat)
    return score_w

def multi_class_auc(y_hat, y):
    y_binary = label_binarize(y, classes=np.unique(y))
    y_hat_binary = label_binarize(y_hat, classes=np.unique(y))

    n_classes = y_binary.shape[1]

    class_auc = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_binary[:, i], y_hat_binary[:, i])
        class_auc.append(auc(fpr, tpr))

    multi_class_auc = np.mean(class_auc)

    return multi_class_auc

# Create an average loss meter
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0

    def update(self, val, n=1):
        self.value += val * n
        self.count += n

    def average(self):
        return self.value / self.count if self.count > 0 else 0

def calculate_metrics(model, loader, task=-1):
    """
    Calculates all metrics for engagement and wandering
    """
    model.eval()
    y_hat = []
    y = []
    val_loss = AverageMeter()
    for data in loader:
        inputs = data['features'].float()
        labels = data['labels']

        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            preds = model(inputs)

        if task == -1:
            y_w, y_e = preds
            y_e = y_e.argmax(dim=1).unsqueeze(1)
            y_w = y_w.argmax(dim=1).unsqueeze(1)

            _y_hat = torch.cat([y_w.cpu(), y_e.cpu()], dim=-1).tolist()
        else:
            _y_hat = preds.argmax(dim=1).cpu().tolist()

        _y = labels.cpu().tolist()
        
        y_hat.extend(_y_hat)
        y.extend(_y)
    
    y_hat = np.array(y_hat)
    y = np.array(y)

    if task == -1:
        return {
            'acc': {
                    "wander": calculate_accuracy(y_hat[:, 0], y[:, 0]),
                    "engage": calculate_accuracy(y_hat[:, 1], y[:, 1]) 
            },
            'auc': {
                'wander': binary_class_auc(y_hat[:, 0], y[:, 0]),
                'engage': multi_class_auc(y_hat[:, 1], y[:, 1])

            },
            'f1': {
                'wander': calculate_f1_binary(y_hat[:, 0], y[:, 0]),
                'engage': calculate_f1_multiclass(y_hat[:, 1], y[:, 1])

            },
            'class_acc': {
                'wander': calculate_class_wise_accuracy(y_hat[:, 0], y[:, 0]),
                'engage': calculate_class_wise_accuracy(y_hat[:, 1], y[:, 1])

            },
            'weighted_f1':{
                'wander': calculate_weighted_f1(y_hat[:, 0], y[:, 0]),
                'engage': calculate_weighted_f1(y_hat[:, 1], y[:, 1])
            },
            'weighted_precision':{
                'wander': calculate_weighted_precision(y_hat[:, 0], y[:, 0]),
                'engage': calculate_weighted_precision(y_hat[:, 1], y[:, 1])
            },
            'weighted_recall':{
                'wander': calculate_weighted_recall(y_hat[:, 0], y[:, 0]),
                'engage': calculate_weighted_recall(y_hat[:, 1], y[:, 1])
            },
            'report': {
                'wander': report(y_hat[:, 0], y[:, 0]),
                'engage': report(y_hat[:, 1], y[:, 1])
            }
        }
    else:
        res = {
            'acc': calculate_accuracy(y_hat, y[:, task]),
            'class_acc': calculate_class_wise_accuracy(y_hat, y[:, task]),
            'report': report(y_hat, y[:, task]),
            'loss': val_loss.average(),
            'weighted_f1': calculate_weighted_f1(y_hat, y[:, task]),
            'weighted_precision': calculate_weighted_precision(y_hat, y[:, task]),
            'weighted_recall': calculate_weighted_recall(y_hat, y[:, task])

        }
        if task == 0:
            res['f1'] = calculate_f1_binary(y_hat, y[:, task])
            res['auc'] = binary_class_auc(y_hat, y[:, task])
        else:
            res['f1'] = calculate_f1_multiclass(y_hat, y[:, task])
            res['auc'] = multi_class_auc(y_hat, y[:, task])

        return res