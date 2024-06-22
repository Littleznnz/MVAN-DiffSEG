import glob
import os
import pickle
import random
import time
import math
import logging
import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm
import os
import features
import model
import data_loader

# 设置指定使用第 n 块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
setup_seed(987654)
attention_head = 4
attention_hidden = 64
# import CapsNet

learning_rate = 0.001
Epochs = 201
BATCH_SIZE = 32

T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neu': 1,  # 1099
    # 'neutral': 1,  # 1099
    # 'happy': 1,
    'hap': 1,
    'sad': 1,
    # 'angry': 1,
    'ang': 1,
    'fea':1,
    'dis':1,
    'bored':1,
}
# FEATURES_TO_USE = 'melspectrogram'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
featuresExist = False
# featuresExist = True
impro_or_script = 'impro'
# featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
featuresFileName = 'features_{}_{}_EMODB_7emo.pkl'.format(FEATURES_TO_USE, impro_or_script)
toSaveFeatures = True
# toSaveFeatures = False
# WAV_PATH = "./data/"
# WAV_PATH = "F:/IEMOCAP_full_release/data/"
# WAV_PATH = r"E:\data\IEMOCAP\path_to_wavs2"CASIA_sort
WAV_PATH = r"E:\head_fusion-master\EmoDB_sort\EmoDB_sort"
RATE = 16000
# MODEL_NAME = 'MyModel_2'
# MODEL_PATH = 'models/{}_{}.pth'.format(MODEL_NAME, FEATURES_TO_USE)

dict = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
    'fea': torch.Tensor([4]),
    'sur': torch.Tensor([5]),
    'bored':torch.Tensor([6]),
}
label_num = {
    'neutral': 0,
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'fea': 0,
    'sur': 0,
    'bored': 0,
}


def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000, spec=False):
    path = path.rstrip('/')
    wav_files = glob.glob(path+'/*' + '/*.wav')
    meta_dict = {}
    val_dict = {}
    LABEL_DICT1 = {
        # '01': 'neutral',
        '01': 'neu',
        # '02': 'frustration',
        # '03': 'happy',
        '04': 'sad',
        # '05': 'angry',
        '05': 'ang',
        '06': 'fea',
        # '07': 'happy',  # excitement->happy
        '07': 'hap',  # excitement->happy
        '08': 'dis',
        '09': 'bored'
    }

    label_num = {
        # 'neutral': 0,
        'neu': 0,
        # 'happy': 0,
        'hap': 0,
        'sad': 0,
        # 'angry': 0,
        'ang': 0,
        'fea': 0,
        'dis': 0,
        'bored': 0,
    }

    train_files = []
    valid_files = []
    wav_files_spec = []
    if spec == True:

        # 设置四种情绪为1099条
        neu_file = []
        hap_file = []
        sad_file = []
        ang_file = []
        fea_file = []
        sur_file = []

        neu_file_spec = []
        hap_file_spec = []
        sad_file_spec = []
        ang_file_spec = []
        for wav_file in wav_files:
            label = str(os.path.basename(wav_file).split('-')[2])
            if label == '01':  # neutral  1099
                neu_file.append(wav_file)
            elif label == '04':  # sad 608
                sad_file.append(wav_file)
            elif label == '05':  # angry 289
                ang_file.append(wav_file)
            elif label == '06':  # angry 289
                fea_file.append(wav_file)
            elif label == '08':  # angry 289
                sur_file.append(wav_file)
            else:  # 07  happy 663
                hap_file.append(wav_file)

        hap_indices = list(np.random.choice(range(len(hap_file)), 436, replace=False))
        sad_indices = list(np.random.choice(range(len(sad_file)), 491, replace=False))
        ang_indices = list(np.random.choice(range(len(ang_file)), 232, replace=False))

        for i in hap_indices:
            hap_file_spec.append(hap_file[i])
        for i in sad_indices:
            sad_file_spec.append(sad_file[i])
        ang_file_spec = np.tile(ang_file, 2).tolist()
        for i in ang_indices:
            ang_file_spec.append(ang_file[i])

        # print(len(hap_file),len(sad_file),len(ang_file1))
        wav_files_spec = hap_file_spec + sad_file_spec + ang_file_spec
        wav_files = wav_files + wav_files_spec
    n = len(wav_files)


    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    for i in train_indices:
        train_files.append(wav_files[i])
    for i in valid_indices:
        valid_files.append(wav_files[i])

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(train_files)):
        label = str(wav_file.split('\\')[4])
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            wav_data = np.pad(wav_data,(0,t * RATE-len(wav_data)+1))
            # continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            assert t - train_overlap > 0
            index += int((t - train_overlap) * RATE / overlapTime[label])
            label_num[label] += 1
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    train_X = []
    train_y = []
    for k in meta_dict:
        train_X.append(meta_dict[k]['X'])
        train_y += meta_dict[k]['y']

    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
        train_X.shape, train_y.shape)

    if (val_overlap >= t):
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        # label = str(os.path.basename(wav_file).split('-')[2])
        # if (label not in LABEL_DICT1):
        #     continue
        # if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
        #     continue
        # label = LABEL_DICT1[label]
        label = str(wav_file.split('\\')[4])
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            wav_data = np.pad(wav_data, (0, t * RATE - len(wav_data)+1))
            # continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    return train_X, train_y, val_dict


def process_features(X, u=255):
    X = torch.from_numpy(X)
    max = X.max()
    X = X / max
    X = X.float()
    X = torch.sign(X) * (torch.log(1 + u * torch.abs(X)) / torch.log(torch.Tensor([1 + u])))
    X = X.numpy()
    return X


def plot_confusion_matrix(actual, predicted, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    predicted = confusion_matrix(actual, predicted)
    # predicted = confusion_matrix(actual, predicted, labels)
    cm = predicted.astype('float') / predicted.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(labels) - 0.5, -0.5)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':

    # logging.info("creating meta dict...")

    train_X, train_y, val_dict = process_data(WAV_PATH, t=T_stride, train_overlap=T_overlop, spec=False)

    print(train_X.shape)
    print(len(val_dict))

    # train_X_spec, train_y_spec, val_dict_spec = process_data(WAV_PATH, t=T_stride, train_overlap=T_overlop,
    #                                                          spec=False)
    # print(train_X_spec.shape)
    # print(len(val_dict_spec))

    print("getting features")
    logging.info('getting features')
    print("未增强数据")
    feature_extractor = features.FeatureExtractor(rate=RATE, spec=True)
    train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
    valid_features_dict = {}

    for _, i in enumerate(val_dict):
        X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
        valid_features_dict[i] = {
            'X': X1,
            'y': val_dict[i]['y']
        }
    # spec 拓展数据
    # print("增强数据")
    # feature_extractor = features.FeatureExtractor(rate=RATE, spec=True)
    # train_X_features_spec = feature_extractor.get_features(FEATURES_TO_USE, train_X_spec)

    # valid_features_dict_spec = {}
    # for _, i in enumerate(val_dict_spec):
    #     X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict_spec[i]['X'])
    #     valid_features_dict_spec[i] = {
    #         'X': X1,
    #         'y': val_dict_spec[i]['y']
    #     }
    # train_X_features_sum = np.vstack((train_X_features, train_X_features_spec))
    # train_y_sum = np.hstack((train_y, train_y_spec))
    # # valid_features_dict_sum = {**valid_features_dict,**valid_features_dict_spec}
    # valid_features_dict_sum = list(valid_features_dict.items()) + list(valid_features_dict_spec.items())
    # valid_features_dict.update(valid_features_dict_spec)
    # if toSaveFeatures == True:
    #     features = {'train_X': train_X_features_sum, 'train_y': train_y_sum,
    #                 'val_dict': valid_features_dict_sum}
    if toSaveFeatures == True:
        features = {'train_X': train_X_features, 'train_y': train_y,
                    'val_dict': valid_features_dict}
    with open(featuresFileName, 'wb') as f:
        pickle.dump(features, f)
