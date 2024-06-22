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
from scipy.spatial.distance import pdist
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# import librosa
from tqdm import tqdm
import os
# import features
import net
import data_loader
import pandas as pd
from net import Unet,GaussianDiffusion,GaussianDiffusionBlock,Generator_Dm,Discriminator_Dm,Encoder_Dm,Classifiar_Dm #从网络中找到扩散模型定义的块
from sklearn.model_selection import train_test_split
import math
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置指定使用第 n 块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(987655)
attention_heads = 4
attention_hidden = 64

learning_rate = 0.001
Epochs = 6
BATCH_SIZE = 16

T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neu': 1,
    'hap': 1,
    'sad': 1,
    'ang': 1,
}
FEATURES_TO_USE = 'mfcc'
impro_or_script = 'impro'
featuresFileName = 'E:/pytorch_project/speech_emotion_recognition/taslp/features4_{}_{}_CASIA.pkl'.format(FEATURES_TO_USE, impro_or_script)
# featuresFileName12 = './taslp/features_{}_{}_12channel.pkl'.format(FEATURES_TO_USE, impro_or_script)
# featuresFileName13 = './taslp/features_{}_{}_13channel.pkl'.format(FEATURES_TO_USE, impro_or_script)
WAV_PATH =  "E:/pytorch_project/speech_emotion_recognition/interspeech2023/interspeech21_emotion/path_to_wavs"
RATE = 16000
tripletData = 'E:/pytorch_project/speech_emotion_recognition/taslp/triplet3_3channel_CASIA.pkl'
# tripletData12 = 'triplet_12channel.pkl'
# tripletData13 = 'triplet_13channel.pkl'

dict = {
    'neu': torch.Tensor([0]),
    'hap': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'ang': torch.Tensor([3]),
}
label_num = {
    'neu': 0,
    'hap': 0,
    'sad': 0,
    'ang': 0,
}


def plot_confusion_matrix(actual, predicted, labels, epochs_data , cmap=plt.cm.Blues):
    title='第{}个Epoch的混淆矩阵'.format(epochs_data+1)
    predicted = confusion_matrix(actual, predicted)
    # predicted = confusion_matrix(actual, predicted, labels)
    cm = predicted.astype('float') / predicted.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print("confusion_matrix:",cm)
    plt.clf()
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
    os.makedirs("image", exist_ok=True)  # 创建目录，如果目录已存在则不报错
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在的绝对路径
    save_path = os.path.join(script_dir, "image", "pit_{}.jpg".format(epochs_data + 1))
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':

    with open(featuresFileName, 'rb')as f:
        features = pickle.load(f)
    train_X_features = features['train_X']
    train_y = features['train_y']  # ang: 704 neu: 2290 hap: 1271 sad: 1592
                                   # ang: 2835 neu: 3682 hap: 4302 sad: 3546
    valid_features_dict = features['val_dict']

    # for _, i in enumerate(valid_features_dict):
    #     x = valid_features_dict[i]['X']
    #     x1 = x[:, 0, :, :]
    #     # x2 = x[:, 1, :, :]
    #     x3 = x[:, 2, :, :]
    #     # x12 = np.stack((x1, x2), axis=1)
    #     x13 = np.stack((x1, x3), axis=1)
    #     # valid_features_dict[i]['X'] = x12
    #     valid_features_dict[i]['X'] = x13

    # features12 = {'train_X': train_X_features, 'train_y': train_y,
    #             'val_dict': valid_features_dict}
    # with open(featuresFileName12, 'wb') as f:
    #     pickle.dump(features12, f)

    # features13 = {'train_X': train_X_features, 'train_y': train_y,
    #               'val_dict': valid_features_dict}
    # with open(featuresFileName13, 'wb') as f:
    #     pickle.dump(features13, f)

    with open(tripletData, 'rb')as f:
        tripletlossData = pickle.load(f)
    anchor = tripletlossData['anchor']
    positive = tripletlossData['positive']
    negative = tripletlossData['negative']
    anchor_label = tripletlossData['anchor_label']
    pos_label = tripletlossData['pos_label']
    neg_label = tripletlossData['neg_label']

    # anchor1 = anchor[:, 0, :, :]
    # anchor2 = anchor[:, 1, :, :]
    # anchor3 = anchor[:, 2, :, :]
    # anchor12 = np.stack((anchor1, anchor2), axis=1)
    # anchor13 = np.stack((anchor1, anchor3), axis=1)
    #
    # positive1 = anchor[:, 0, :, :]
    # positive2 = anchor[:, 1, :, :]
    # positive3 = anchor[:, 2, :, :]
    # positive12 = np.stack((positive1, positive2), axis=1)
    # positive13 = np.stack((positive1, positive3), axis=1)
    #
    # negative1 = anchor[:, 0, :, :]
    # negative2 = anchor[:, 1, :, :]
    # negative3 = anchor[:, 2, :, :]
    # negative12 = np.stack((negative1, negative2), axis=1)
    # negative13 = np.stack((negative1, negative3), axis=1)
    #

    #
    # features12 = {'anchor': anchor12, 'positive': positive12, 'negative': negative12,
    #             'anchor_label': anchor_label, 'pos_label': pos_label, 'neg_label': neg_label, }
    # with open(tripletData12, 'wb') as f:
    #     pickle.dump(features12, f)
    #
    # features13 = {'anchor': anchor13, 'positive': positive13, 'negative': negative13,
    #             'anchor_label': anchor_label, 'pos_label': pos_label, 'neg_label': neg_label, }
    # with open(tripletData13, 'wb') as f:
    #     pickle.dump(features13, f)

    with open('fake_label.pkl', 'rb')as f:
        fake_label = pickle.load(f)

    train_data = data_loader.TripletDataSet_fake(anchor, positive, negative, anchor_label, pos_label, neg_label,
                                                 fake_label)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    #使用常规的GAN网络部分
    # F_net = net.Encoder1(attention_heads, attention_hidden)  #Encoder1为修改attention后的编码器，Encoder为原始版本，Encoder2为适应扩散模型修改的编码器
    # D_net = net.Discriminator1()
    # G_net = net.Generator1()
    # Classifiar = net.Classifiar()

    #使用扩散模型部分
    channels = 16
    input_channel = 4
    image_size = 8
    model_unet = Unet(dim=channels, channels=input_channel)
    time_step = 100
    timesteps = 101
    F_net = Encoder_Dm(attention_heads,attention_hidden)  # Encoder1为修改attention后的编码器，Encoder为原始版本，Encoder2为适应扩散模型修改的编码器
    # D_net = net.Discriminator1()
    # G_net = net.Generator1()
    diffusion_block = GaussianDiffusionBlock(model_unet, image_size, timesteps)
    G_net = Generator_Dm(model_unet, image_size, timesteps)
    D_net = Discriminator_Dm()
    Classifiar = Classifiar_Dm()


    # TripletLoss = net.TripletLoss()
    if torch.cuda.is_available():
        F_net = F_net.cuda()
        D_net = D_net.cuda()
        G_net = G_net.cuda()
        Classifiar = Classifiar.cuda()
        # TripletLoss = TripletLoss.cuda()

    softmax_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    triplet_criterion = nn.TripletMarginLoss()

    optimizer_F = optim.Adam(F_net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)
    optimizer_D = optim.Adam(D_net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)
    optimizer_G = optim.Adam(G_net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)
    optimizer_classifiar = optim.Adam(Classifiar.parameters(), lr=learning_rate,
                                      weight_decay=1e-6)
    logging.info("training...")
    maxWA = 0
    maxUA = 0

    final_labels = ['NEU', 'HAP', 'SAD', 'ANG']
    label_to_class = {0: 'ANG', 1: 'SAD', 2: 'NEU', 3: 'HAP'}

    totalrunningTime = 0
    loss_all = []
    Acc_all = []
    UA_all = []
    UA_mean_all = []
    eporch_labels = []  # 保存学的最好的轮次
    for i in range(Epochs):
        epochs_data = i
        pred_model = []
        actual = []
        startTime = time.perf_counter()
        tq = tqdm(total=len(anchor_label))
        F_net.train()
        D_net.train()
        G_net.train()
        Classifiar.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            x_anchor, x_pos, x_neg, y_pos, y_neg, fake_label = data
            if torch.cuda.is_available():
                x_anchor = x_anchor.cuda()
                x_pos = x_pos.cuda()
                x_neg = x_neg.cuda()
                y_pos = y_pos.cuda()
                y_neg = y_neg.cuda()
                fake = torch.zeros((y_pos.shape[0], 1)).cuda()
                real = torch.ones((y_pos.shape[0], 1)).cuda()

            F_anchor = F_net(x_anchor)
            F_positive = F_net(x_pos)
            F_negative = F_net(x_neg)

            #原始GAN部分的生成器输出
            # G_F_anchor = G_net(F_anchor)  #G_F_anchor,G_F_positive,G_F_negative
            # G_F_positive = G_net(F_positive)
            # G_F_negative = G_net(F_negative)

            #扩散模型部分的生成器输出
            G_F_anchor, _ = G_net(F_anchor, time_step)
            G_F_positive, _ = G_net(F_positive, time_step)
            G_F_negative, _ = G_net(F_negative, time_step)

            D_G_F_anchor, D_G_F_anchor4 = D_net(G_F_anchor)  #D_F_anchor,D_F_positive,D_F_negative,D_G_F_anchor,D_G_F_positive,D_G_F_negative
            D_G_F_positive, D_G_F_positive4 = D_net(G_F_positive) #D_F_anchor4,D_F_positive4,D_F_negative4,D_G_F_anchor4,D_G_F_positive4,D_G_F_negative4
            D_G_F_negative, D_G_F_negative4 = D_net(G_F_negative)

            D_F_anchor, D_F_anchor4 = D_net(F_anchor)        #D_F_anchor,D_F_positive,D_F_negative
            D_F_positive, D_F_positive4 = D_net(F_positive)
            D_F_negative, D_F_negative4 = D_net(F_negative)

            anchor_out = Classifiar(F_anchor)
            positive_out = Classifiar(F_positive)
            negative_out = Classifiar(F_negative)

            F_tri_loss = triplet_criterion(G_F_anchor, G_F_positive, G_F_negative)
            G_tri_loss = triplet_criterion(G_F_anchor, G_F_negative, G_F_positive)

            DREALLOSS = (bce_criterion(D_F_anchor.squeeze(1), real.squeeze(1)) +
                         bce_criterion(D_F_positive.squeeze(1),real.squeeze(1)) +
                         bce_criterion(D_F_negative.squeeze(1),real.squeeze(1))) / 3

            DREALLOSS1 = (softmax_criterion(D_F_anchor4.squeeze(1), y_pos.squeeze(1)) +
                          softmax_criterion(D_F_positive4.squeeze(1),y_pos.squeeze(1)) +
                          softmax_criterion(D_F_negative4.squeeze(1),y_neg.squeeze(1))) / 3

            DFAKELOSS = (bce_criterion(D_G_F_anchor.squeeze(1), fake.squeeze(1)) +
                         bce_criterion(D_G_F_positive.squeeze(1),fake.squeeze(1)) +
                         bce_criterion(D_G_F_negative.squeeze(1),fake.squeeze(1))) / 3

            GCLSLOSS = (softmax_criterion(D_G_F_anchor4, y_pos.squeeze(1)) +
                        softmax_criterion(D_G_F_positive4,y_pos.squeeze(1))+
                        softmax_criterion(D_G_F_negative4,y_neg.squeeze(1))) / 3

            Classifiar_loss = (softmax_criterion(anchor_out, y_pos.squeeze(1)) +
                               softmax_criterion(positive_out,y_pos.squeeze(1)) +
                               softmax_criterion(negative_out,y_neg.squeeze(1))) / 3  #分类网咯损失

            F_loss = F_tri_loss + Classifiar_loss  # encode 主干网络
            G_loss = G_tri_loss + GCLSLOSS         # generator 生成器
            D_loss = DREALLOSS + DFAKELOSS + DREALLOSS1 # discriminator 辨别器
            classlabel = Classifiar_loss + GCLSLOSS  # 分类器损失

            optimizer_F.zero_grad()
            optimizer_D.zero_grad()
            optimizer_classifiar.zero_grad()
            optimizer_G.zero_grad()

            F_loss.backward(retain_graph=True)
            D_loss.backward(retain_graph=True)
            classlabel.backward(retain_graph=True)
            G_loss.backward()

            optimizer_F.step()
            optimizer_D.step()
            optimizer_classifiar.step()
            optimizer_G.step()

            loss = classlabel + F_loss + G_loss + D_loss
            print_loss += loss.data.item() * BATCH_SIZE

            tq.update(BATCH_SIZE)
        tq.close()
        print('epoch: {}, loss: {:.4}'.format(i, print_loss / len(train_X_features)))
        logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
        loss_all.append(print_loss / len(train_X_features))
        if (i > 0 and i % 10 == 0):
            learning_rate = learning_rate / 10
            for param_group in optimizer_F.param_groups:
                param_group['lr'] = learning_rate
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = learning_rate
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = learning_rate
            for param_group in optimizer_classifiar.param_groups:
                param_group['lr'] = learning_rate
        # validation
        endTime = time.perf_counter()
        # endTime = time.clock()
        totalrunningTime += endTime - startTime
        print("totalrunningTime：",totalrunningTime)

        F_net.eval()
        D_net.eval()
        G_net.eval()
        Classifiar.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)
        for _, i in enumerate(valid_features_dict):
            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = torch.from_numpy(x).float()
            y = dict[y[0]].long()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # if (x.shape[0] != 16):
            #     x=x.permute(1,0,2,3)
            #     conv_two = nn.Conv2d(kernel_size=(1, 1), in_channels=x.shape[1], out_channels=16)
            #     conv_two.cuda()
            #     x=conv_two(x)
            #     x = x.reshape(-1,3,26,63)


            # if x.shape[0] == 16:
            out = F_net(x)
            out = Classifiar(out)
            pred = torch.Tensor([0, 0, 0, 0]).cuda()
            for j in range(out.size(0)):
                pred += out[j]
            pred = pred / out.size(0)
            pred = torch.max(pred, 0)[1]

            pred_model.append(pred.cpu().numpy())
            actual.append(y.data.cpu().numpy())
            # else:
            #     print("断开了")
            #     continue

            if (pred == y):
                num_correct += 1
            matrix[int(y), int(pred)] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(valid_features_dict)
        if (maxWA < WA):
            maxWA = WA
            torch.save(F_net, r'net.pkl')  # 保存整个神经网络到net1.pkl中
            torch.save(F_net.state_dict(), r'best_model_weights.pth')  # 保存网络里的参数
            eporch_labels.append(epochs_data)

        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4

        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA:{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        Acc_all.append(WA)
        UA_all.append(UA)
        UA_mean_all.append(sum(UA) / 4)
        logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA:{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        print(matrix)
        logging.info(matrix)
        pred_model = tuple(pred_model)
        pred_model = np.hstack(pred_model)
        actual = tuple(actual)
        actual = np.hstack(actual)
        pred_with_label = [label_to_class[label] for label in list(pred_model)]
        actual_with_label = [label_to_class[label] for label in list(actual)]
        plot_confusion_matrix(actual_with_label, pred_with_label, final_labels,epochs_data)
        print('\n Classification Report \n {} \n'.format(classification_report(actual_with_label, pred_with_label)))
    # torch.save(model, r'net.pkl')  # 保存整个神经网络到net1.pkl中
    # torch.save(model.state_dict(), r'net_params.pkl')  # 保存网络里的参数
    # 创建一个字典，将列表作为值，键作为列名
    data = {
        'loss': loss_all,
        'accuracy': Acc_all,
        'UA': UA_all,
        'UA_mean': UA_mean_all
    }
    df = pd.DataFrame(data)  # 使用字典创建数据框
    df.to_excel('data.xlsx', index=False)  # 保存训练数据至excel中
    print("模型训练的最好轮次为：", eporch_labels[-1])
