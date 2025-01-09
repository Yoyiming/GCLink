import scipy.sparse as sp
import pandas as pd
import numpy as np
import random
import glob
import os
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
from PytorchTools import EarlyStopping

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from torch_geometric.utils import negative_sampling
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3,
                    help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default=20, help='Number of epoch.')
parser.add_argument('-num_head', type=list,
                    default=[3, 3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dim', type=int,
                    default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16,
                    help='The dimension of latent layer')
parser.add_argument('-batch_size', type=int, default=256,
                    help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False,
                    help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=8, help='Random seed')
parser.add_argument('-Type', type=str, default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False,
                    help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction', type=str, default='concate',
                    help='how to integrate multihead attention')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class My_Model(nn.Module):
    def __init__(self, encoder_model1, encoder_model2):
        super(My_Model, self).__init__()
        self.encoder1 = encoder_model1
        self.encoder2 = encoder_model2

    def forward(self, data_feature, adj, train_data):
        index = adj.coalesce().indices()  # 获取adj的indices
        v = adj.coalesce().values()       # 获取index对应的value
        size = adj.coalesce().size()      # 获取adj的行数和列数

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder1(data_feature, adj1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder2(data_feature, adj2, train_data)

        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def autoencoder_train(feature, hidden_dim, auto_epochs=50):
    input_dim = feature.shape[1]
    autoencoder = Autoencoder(input_dim, hidden_dim)
    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = MyDataset(feature)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    autoencoder.train()
    for epoch in range(auto_epochs):
        for data in data_loader:
            optimizer.zero_grad()

            # 前向传播和计算损失
            encoded, outputs = autoencoder(data)
            loss = criterion(outputs, data)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, auto_epochs, loss.item()))

    # 使用训练好的自编码器进行降维
    feature = torch.tensor(feature)
    autoencoder.eval()
    with torch.no_grad():
        encoded_data, _ = autoencoder(feature)
    encoded_data = np.array(encoded_data)
    return encoded_data


def gae_train(data_feature, adj1, adj2, gae_model1, gae_model2, optimizer1, optimizer2, scheduler1, scheduler2):

    pretrain_loss1 = 0.0
    pretrain_loss2 = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        gae_model1.train()
        optimizer1.zero_grad()

        gae_model2.train()
        optimizer2.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        z1, train_tf1, train_target1, pred1 = gae_model1(data_feature, adj1, train_x)
        z2, train_tf2, train_target2, pred2 = gae_model2(data_feature, adj2, train_x)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss1 = F.binary_cross_entropy(pred1, train_y)
        loss2 = F.binary_cross_entropy(pred2, train_y)

        loss1.backward(retain_graph=True)
        optimizer1.step()
        scheduler1.step()

        loss2.backward(retain_graph=True)
        optimizer2.step()
        scheduler2.step()

        pretrain_loss1 += loss1.item()  #item()用于获取标量值
        pretrain_loss2 += loss2.item()

    return float(pretrain_loss1), float(pretrain_loss2), pred1, pred2


# @torch.no_grad()
def gae_test(test_data, adj1, adj2, gae_model1, gae_model2):
    gae_model1.eval()
    z1 = gae_model1(test_data, adj1)
    index1 = adj1.coalesce().indices()  # 获取adj的indices

    gae_model2.eval()
    z2 = gae_model2(test_data, adj2)
    # index2 = adj2.coalesce().indices()  # 获取adj的indices

    from sklearn.metrics import average_precision_score, roc_auc_score
    neg_edge_index = negative_sampling(index1, z1.size(0))
    pos_y1 = z1.new_ones(index1.size(1))
    neg_y1 = z1.new_zeros(neg_edge_index.size(1))
    y1 = torch.cat([pos_y1, neg_y1], dim=0)
    pos_pred1 = torch.sigmoid((z1[index1[0]] * z1[index1[1]]).sum(dim=1))
    neg_pred1 = torch.sigmoid((z1[neg_edge_index[0]] * z1[neg_edge_index[1]]).sum(dim=1))
    pred1 = torch.cat([pos_pred1, neg_pred1], dim=0)
    y1, pred1 = y1.detach().cpu().numpy(), pred1.detach().cpu().numpy()

    # pos_y2 = z2.new_ones(index1.size(1))
    # neg_y2 = z2.new_zeros(neg_edge_index.size(1))
    # y2 = torch.cat([pos_y2, neg_y2], dim=0)
    pos_pred2 = torch.sigmoid((z2[index1[0]] * z2[index1[1]]).sum(dim=1))
    neg_pred2 = torch.sigmoid((z2[neg_edge_index[0]] * z2[neg_edge_index[1]]).sum(dim=1))
    pred2 = torch.cat([pos_pred2, neg_pred2], dim=0)
    pred2 = pred2.detach().cpu().numpy()

    return roc_auc_score(y1, pred1), average_precision_score(y1, pred1), roc_auc_score(y1, pred2), average_precision_score(y1, pred2)


def pretrain(data_feature, adj, model1, model2, epochs=20):
    # losses = []
    # train_aucs = []
    # train_aps = []

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.99)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.99)

    index = adj.coalesce().indices()  # 获取adj的indices
    v = adj.coalesce().values()
    size = adj.coalesce().size()

    for epoch in range(1, epochs + 1):

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        loss1, loss2, pred1, pred2 = gae_train(data_feature, adj1, adj2, model1, model2, optimizer1, optimizer2, scheduler1, scheduler2)
        # losses.append(loss1)
        # model1.eval()
        # model2.eval()
        # AUC1, AUPR1, _ = Evaluation(y_pred=pred1, y_true=validation_data[:, -1], flag=args.flag)
        # AUC2, AUPR2, _ = Evaluation(y_pred=pred2, y_true=validation_data[:, -1], flag=args.flag)
        print('Epoch:{}'.format(epoch),
              'loss1:{:.5F}'.format(loss1),
              'loss2:{:.5F}'.format(loss2))


def train(model, contrast_model, optimizer):

    running_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2 = model(data_feature, adj, train_x)  # 这里实际上不需要tf_embed和target_embed

        con_loss = contrast_model(h1=embed1, h2=embed2)
        # con_loss = contrast_model(g1=tf_embed1, g2=tf_embed2, batch=args.batch_size)
        # con_loss += contrast_model(g1=target_embed1, g2=target_embed2, batch=args.batch_size)

        # index = adj.coalesce().indices()  # 获取adj的indices
        # Recon_loss1 = recon_loss(embed1, pos_edge_index=index, neg_edge_index=None)
        # Recon_loss2 = recon_loss(embed2, pos_edge_index=index, neg_edge_index=None)

        # con_loss = contrast_model(tf_embed1, tf_embed2)
        # con_loss += contrast_model(target_embed1, target_embed2)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss_BCE1 = F.binary_cross_entropy(pred1, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2, train_y)
        loss = loss_BCE1 + loss_BCE2 + 0.5 * con_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        # print('Recon_loss:{:.5F}'.format(Recon_loss),
        #       'lossBCE2:{:.5F}'.format(loss_BCE2),
        #       'con_loss:{:.3F}'.format(con_loss))
    return running_loss


exp_file = 'Specific Dataset/hESC/TFs+1000/BL--ExpressionData.csv'
tf_file = 'Specific Dataset/hESC/TFs+1000/TF.csv'
target_file = 'Specific Dataset/hESC/TFs+1000/Target.csv'

train_file = 'Data/Specific/hESC 1000/sample1/Train_set.csv'
test_file = 'Data/Specific/hESC 1000/sample1/Test_set.csv'
val_file = 'Data/Specific/hESC 1000/sample1/Validation_set.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据以及TF和target gene
data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()  # 读取表达数据并进行归一化
print('feature shape ', feature.shape)
tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)

# PCA降维
# pca = PCA(n_components=380)
# data_feature = pca.fit_transform(feature)
# print('pca feature shape ', data_feature.shape)

# SVD分解降维
U, S, V = np.linalg.svd(feature.T)
k = 200
data_feature = V[:k, :].T
print('svd feature shape ', data_feature.shape)

# 自编码器降维
# data_feature = autoencoder_train(feature, 380)
# print('autoencoder feature shape ', data_feature.shape)

# Kernel PCA降维
# kernel_pca = KernelPCA(n_components=380, kernel='rbf')
# data_feature = kernel_pca.fit_transform(feature)
# print('kernel_pca feature shape ', data_feature.shape)

data_feature = torch.from_numpy(data_feature)
tf = torch.from_numpy(tf)
data_feature = data_feature.to(device)
tf = tf.to(device)

# concat成一个大的训练集
train_data = pd.read_csv(train_file, index_col=0)
test_data = pd.read_csv(test_file, index_col=0)
validation_data = pd.read_csv(val_file, index_col=0)

data = pd.concat([train_data, test_data, validation_data])
# x = data.drop('Label', axis=1)
# y = data['Label']
# train_data, validation_data, train_label, valid_label = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
# train_data = pd.concat([train_data, train_label], axis=1).values
# validation_data = pd.concat([validation_data, valid_label], axis=1).values
train_data = data.values
train_load = scRNADataset(train_data, data_feature.shape[0], flag=args.flag)

# 创建邻接矩阵
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2saprse_tensor(adj)

contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)
# contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2G').to(device)
# contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
# print('feature size ', feature.size()[1])
encoder_model1 = GENELink(input_dim=data_feature.size()[1],
                          hidden1_dim=args.hidden_dim[0],
                          hidden2_dim=args.hidden_dim[1],
                          hidden3_dim=args.hidden_dim[2],
                          output_dim=args.output_dim,
                          num_head1=args.num_head[0],
                          num_head2=args.num_head[1],
                          alpha=args.alpha,
                          device=device,
                          type=args.Type,
                          reduction=args.reduction
                          ).to(device)
encoder_model2 = GENELink(input_dim=data_feature.size()[1],
                          hidden1_dim=args.hidden_dim[0],
                          hidden2_dim=args.hidden_dim[1],
                          hidden3_dim=args.hidden_dim[2],
                          output_dim=args.output_dim,
                          num_head1=args.num_head[0],
                          num_head2=args.num_head[1],
                          alpha=args.alpha,
                          device=device,
                          type=args.Type,
                          reduction=args.reduction
                          ).to(device)

adj = adj.to(device)
train_data = torch.from_numpy(train_data)
train_data = train_data.to(device)
# validation_data = torch.from_numpy(validation_data)
# validation_data = validation_data.to(device)

# data augmentation methods
# aug1 = A.NodeDropping(pn=0.1)
# aug2 = A.RandomChoice([A.NodeDropping(pn=0.1), A.FeatureMasking(pf=0.1), A.EdgeRemoving(pe=0.1)], 1)
# aug2 = A.NodeDropping(pn=0.1)
# aug2 = A.FeatureMasking(pf=0.1)
aug1 = A.Identity()
aug2 = A.EdgeRemoving(0.3)

model = My_Model(encoder_model1=encoder_model1, encoder_model2=encoder_model2)
model = model.to(device)

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# 预训练
pre_epochs = 20
if pre_epochs > 0:
    pretrain(data_feature, adj, encoder_model1, encoder_model2, pre_epochs)

# 训练
AUC_Threshold = 0
for epoch in range(args.epochs):

    running_loss = train(model, contrast_model, optimizer)
    model.eval()
    _, _, _, _, _, _, _, score = model(data_feature, adj, train_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(
        y_pred=score, y_true=train_data[:, -1], flag=args.flag)
    print('Epoch:{}'.format(epoch + 1),
          'train loss:{:.5F}'.format(running_loss),
          'AUC:{:.3F}'.format(AUC),
          'AUPR:{:.3F}'.format(AUPR))

    # 保存最优模型
    # if AUC > AUC_Threshold:
    #     AUC_Threshold = AUC
torch.save(model.state_dict(), model_path + '/hESC1000.pkl')
        
