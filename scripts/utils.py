from pandas import read_csv
import numpy as np

zh_dataset = ['XNLI']


def get_num_heads(model):
    if model == 'tiny':
        return '16'
    if model == 'base' or model == 'large':
        return '12'
    assert False, 'Not recognized model'


def get_valid_data(dataset, model):
    if dataset in zh_dataset:
        if model == 'tiny':
            return f'data/zh/{dataset}/dev.inference.tiny.dynamic'
        else:
            return f'data/zh/{dataset}/dev.inference.dynamic'
    else:
        return f'data/{dataset}/dev.inference.dynamic'


def get_valid_labels(dataset):
    zh_dataset = ['XNLI']
    if dataset in zh_dataset:
        return f'data/zh/{dataset}/dev.tsv'
    else:
        return f'data/{dataset}/dev.tsv'


def get_num_labels(dataset):
    if dataset == 'XNLI':
        return '3'
    else:
        return '2'


def convert_to_num(label):
    if label == 'neutral':
        return 2
    if label == 'contradiction':
        return 0
    if label == 'entailment':
        return 1
    assert False


def read_label(filename):
    # Pandas is not able to read QNLI data
    if 'QNLI' in filename:
        labels = list()
        with open(filename) as file:
            for line in file:
                try:
                    _, _, label = line.strip().split('\t')
                    labels.append(int(label))
                except ValueError as e:
                    pass
        return np.asarray(labels)
    data = read_csv(filename, sep='\t')
    if 'XNLI' in filename:
        return data['label'].apply(convert_to_num).to_numpy()
    return data['label'].to_numpy()


def get_default_seq_len(dataset):
    if dataset == 'WNLI':
        return 512
    else:
        return 128


def get_metric(dataset):
    metrics = {'CoLA': matthews_corrcoef,
               'WNLI': simple_accuracy,
               'RTE': simple_accuracy,
               'QQP': simple_accuracy,
               'SST-2': simple_accuracy,
               'QNLI': simple_accuracy,
               'XNLI': simple_accuracy,
              }
    return metrics[dataset]


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc
