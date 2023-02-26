# coding=utf-8
# Author: Jung
# Time: 2022/8/15 10:14

import warnings
warnings.filterwarnings("ignore")
from attributed_nets.draw_heatmap import *
from attributed_nets.t_sne import *
import dgl.function as fn
import torch
import numpy as np
import dgl
from dgl.nn import GraphConv
import pickle as pkl
import torch.nn as nn
import argparse
from scipy import sparse
from sklearn import metrics
import random
import scipy.io as scio
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch import LongTensor
import networkx as nx
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
import pandas as pd
from scipy import sparse
import pickle as pkl
import community

random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

def modularity(adj: np.array, pred: np.array):
    """
    非重叠模块度
    :param adj: 邻接矩阵
    :param pred: 预测社区标签
    :return:
    """
    graph = nx.from_numpy_matrix(adj)
    part = pred.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur



def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def density(k, labels, A):
    communities = np.zeros(shape=(A.shape[0], k)) # n * k
    communities[range(A.shape[0]), labels] = 1
    communities = communities.dot(communities.T)
    row, col = np.diag_indices_from(communities)
    communities[row, col] = 0
    _density = communities * A
    return (_density.sum().sum() / (A.sum().sum())) * 0.5

def calculate_entropy(k, pred_labels, feat):
    """
    :param k: 社区个数
    :param pred_labels: 预测社区
    :param num_nodes: 节点的个数
    :param feat: 节点属性
    :return:
    """
    # 初始化两个矩阵

    num_nodes = feat.shape[0]

    label_assemble = np.zeros(shape=(num_nodes, k))
    label_atts = np.zeros(shape=(k, feat.shape[1]))

    label_assemble[range(num_nodes), pred_labels] = 1
    label_assemble = label_assemble.T

    # 遍历每个社区
    for i in range(k):
        # 如果社区中的值大于0，则获得索引
        node_indx = np.where(label_assemble[i] > 0)
        # 获得索引下的所有属性
        node_feat = feat[node_indx]
        label_atts[i] = node_feat.sum(axis=0) # 向下加和

    __count_attrs = label_atts.sum(axis=1)
    __count_attrs = __count_attrs[:,np.newaxis]
    _tmp = label_atts / __count_attrs
    p = (_tmp) * - (np.log2(_tmp + 1e-10))

    p = p.sum(axis=1)
    label_assemble = label_assemble.sum(axis=1)
    __entropy = (label_assemble / num_nodes) * p
    return __entropy.sum()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='the number of epochs')


    return parser.parse_known_args()

def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

class GCN(nn.Module):
    def __init__(self, feat_dim, hid_dim, k):
        super(GCN, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.conv = GraphConv(feat_dim, hid_dim)
        self.conv2 = GraphConv(hid_dim, k)
        """ You can choice nn.Softmax, nn.Tanh() et al."""
        self.act = nn.ReLU()
    def forward(self, graph, feat):
        """ Notice: the final h need to use activation function """
        h = self.conv(graph, feat)
        h = self.act(h)
        h = self.conv2(graph, h)
        return h

def knn_graph(feat, topk, weight = False, loop = True):
    sim_feat = cosine_similarity(feat)
    sim_matrix = np.zeros(shape=(feat.shape[0], feat.shape[0]))

    inds = []
    for i in range(sim_feat.shape[0]):
        ind = np.argpartition(sim_feat[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    for i, vs in enumerate(inds):
        for v in vs:
            if v == i:
                pass
            else:
                if weight is True:
                    sim_matrix[i][v] = sim_feat[i][v]
                    sim_matrix[v][i] = sim_feat[v][i]
                else:
                    sim_matrix[i][v] = 1
                    sim_matrix[v][i] = 1

    sp_matrix = sparse.csr_matrix(sim_matrix)
    dgl_matrix = dgl.from_scipy(sp_matrix)
    if loop is True:
        dgl_matrix = dgl.add_self_loop(dgl_matrix)
    return dgl_matrix


def load_data(name: str):

    with open("datasets/"+name+".pkl", 'rb') as f:
        data = pkl.load(f)

    # 拓扑增加噪声
    adj = data['topo'].toarray()
    adj = difus_node_topo(adj, adj.shape[0], 0.5)
    adj = sparse.csr_matrix(adj)
    graph = dgl.from_scipy(adj)

    # graph = dgl.from_scipy(data['topo'])  # 数据集中自带自环
    return graph, data['attr'].toarray(), data['label']
stmp = 0
class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size= 16): #(1) 注意力网隐藏层维度
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        _beta = beta.detach().numpy()
        topo_weigh_list = []
        knn_weigh_list = []
        for i in range(_beta.shape[0]):
            topo_weigh = _beta[i][0][0]
            knn_weigh = _beta[i][1][0]
            topo_weigh_list.append(topo_weigh)
            knn_weigh_list.append(knn_weigh)
        np.save("weight/"+"knn_weigh_"+str(stmp), knn_weigh_list)
        np.save("weight/" + "topo_weigh_" + str(stmp), topo_weigh_list)
        return (beta * z).sum(1)

""" rename """
class Ays(nn.Module):


    def __init__(self, graph, kgraph, feat, label):
        super(Ays, self).__init__()
        self.graph = graph
        self.kgraph = kgraph
        self.adj = graph.adjacency_matrix().to_dense()
        self.kadj = kgraph.adjacency_matrix().to_dense()
        self.feat = torch.from_numpy(feat).to(torch.float)
        self.labels = label
        self.num_nodes, self.feat_dim = self.feat.shape
        self.k = len(np.unique(self.labels))
        self.atten = Attention(self.k)
        self.t = self.k  # 主题数 #(2)
        """
            Cora: 200(高NMI） 90（高AC）
            Citeseer: 90(NMI:0.0.320)
            uai2010:0.26+(90)
        """
        hid = 200 #(3) 隐藏层维度
        self.gcn = GCN(self.feat_dim, hid, self.k) # 200的NMI高, 90的AC高
        self.knn_gcn = GCN(self.feat_dim, hid, self.k)
        self.B = self.get_b_of_modularity(self.adj)

        """ activate functions """
        self.relu = nn.ReLU()
        self.log_sig = nn.LogSigmoid()
        self.soft = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.bce_with_log = nn.BCEWithLogitsLoss()


        """ k_means """
        self.cluster_layer = Parameter(torch.Tensor(self.k, self.k))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1


        """ parameters """
        self.topo_par = nn.Parameter(torch.FloatTensor(self.k, self.k))
        # self.feat_par = nn.Parameter(torch.FloatTensor(self.feat_dim, self.k))


        self.feat_par = nn.Parameter(torch.FloatTensor(self.t, self.feat_dim)) # t = 4
        self.topic = nn.Parameter(torch.FloatTensor(self.k, self.t))
        torch.nn.init.xavier_uniform_(self.topo_par)
        torch.nn.init.xavier_uniform_(self.feat_par)
        torch.nn.init.xavier_uniform_(self.topic)

    def get_b_of_modularity(self, A):
        K = 1 / (A.sum().item()) * (A.sum(axis=1).reshape(A.shape[0], 1) @ A.sum(axis=1).reshape(1, A.shape[0]))
        return A - K

    def constraint(self):
        w = self.topo_par.data.clamp(0, 1)
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.topo_par.data = w

        # w = self.feat_par.data.clamp(0, 1)
        # col_sums = w.sum(dim=0)
        # w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        # self.feat_par.data = w

        w = self.topic.data.clamp(0, 1)
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.topic.data = w

    def forward(self):

        h = self.gcn(self.graph, self.feat)

        """ 可选是否共参 """
        h_knn = self.gcn(self.kgraph, self.feat) # 共参
        # h_knn = self.knn_gcn(self.kgraph, self.feat) # 非共参

        h_att = self.atten(torch.stack([h, h_knn], dim=1)) #拓扑、属性融合后的embeddings

        q = self.get_q(self.relu(h_att)) #?

        # emb_feat = self.feat @ self.topic @ self.feat_par #? self.feat  #重建属性 decoder

        emb_feat = h_att @ self.topic @ self.feat_par

        emb_topo = h_att @ self.topo_par @ h_att.t() #重建拓扑，decoder


        return self.soft(h_att), q,  self.soft(emb_feat), self.soft(emb_topo)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def swap_node_attr(g: dgl.DGLGraph, feat, ratio: float):

    """
    以一定的比例交换节点属性
    :param g: dgl.DGLGraph
    :param ratio: 交换比例， [0, 1]
    :return:
    """
    n = g.num_nodes()  # 节点数
    num_swap = int(n * ratio * 0.5)
    for i in range(num_swap):
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if a == b:
            continue
        feat[[a, b], :] = feat[[b, a], :]
    return feat


def difus_node_attr(g: dgl.DGLGraph, feat, ratio_node: float, ratio_feat: float):
    """
    给节点属性新增噪声
    :param g: dgl.DGLGraph
    :param ratio: 交换比例， [0, 1]
    :return:
    """
    n = g.num_nodes()  # 节点数
    node_random = int(n * ratio_node * 0.5)
    feat_random = int(n * ratio_feat * 0.5)
    feat_dim = feat.shape[1] # 节点属性的个数
    for i in range(node_random):
        t_or_f = random.randint(0,1)
        a = random.randint(0, n - 1) # 随机选择节点
        b = [random.randint(0, feat_dim - 1) for i in range(feat_random)]
        feat[a,b] = t_or_f
    return feat

def difus_node_topo(adj, n, ratio_node: float):
    """
    给节点属性新增噪声
    :param g: dgl.DGLGraph
    :param ratio: 交换比例， [0, 1]
    :return:
    """
    node_random = int(n * ratio_node * 0.5)
    for i in range(node_random):
        # t_or_f = random.randint(0,1)
        a = random.randint(0, n - 1) # 随机选择节点
        b = random.randint(0, n - 1) # 随机选择节点
        adj[a,b] = 1
        adj[b,a] = 1
    return adj

if __name__ == "__main__":
    args, _ = parse_args()
    printConfig(args)
    graph, feat, label = load_data(args.dataset)
    #feat = difus_node_attr(graph, feat, 0.5, 0.01) # 选多少比例的节点; 选多少比例的属性
    # feat = swap_node_attr(graph, feat, 0.5) #交换一定比例节点的属性
    kgraph = knn_graph(feat, 20) #(4) 近邻数

    model = Ays(graph, kgraph, feat, label)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with torch.no_grad():
        emb = model.gcn(graph, model.feat)
    kmeans = KMeans(n_clusters=model.k, n_init=20) #20
    y_pred = kmeans.fit_predict(emb.data.numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_)

    for epoch in range(args.epochs):

        stmp = epoch

        model.train()
        optimizer.zero_grad()

        if epoch % 5 == 0:
            emb, q, emb_feat, emb_topo = model()
            q = q.detach().data
            p = model.target_distribution(q)
        # old:nmi: 0.540, f1_score=0.227,  ac = 0.243, ari= 0.453
        emb, q, emb_feat, emb_topo = model()
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        feat_loss = F.mse_loss(model.feat, emb_feat) #? 属性损失？

        att_loss = F.mse_loss(model.adj, emb_topo) #拓扑损失
        att_loss2 = F.mse_loss(model.kadj, emb_topo)

        modu_loss = torch.trace(emb.t() @ model.B @ emb) #模块度


        loss =0*kl_loss + feat_loss +  att_loss + 0*att_loss2- modu_loss*0.5

        #visualization(epoch, emb, label)
        run_heatmap(epoch, model.topic.detach().numpy(), model.k)

        model.eval()
        pred = emb.argmax(dim=1)

        nmi = compute_nmi(pred, model.labels)
        ac = compute_ac(pred, model.labels)
        f1 = computer_f1(pred, model.labels)
        ari = computer_ari(model.labels, pred)
        mody_org = modularity(model.adj.numpy(), pred)
        # dety = density(model.k, pred.detach().numpy(), model.adj.numpy())
        etp = calculate_entropy(model.k, pred.detach().numpy(), model.feat.numpy())
        print(
            'epoch={}, loss={:.3f},  nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f},  mody_org = {:.3f}'.format(
                epoch,
                loss,
                nmi,
                f1,
                ac,
                ari,
                mody_org,
            ))
        loss.backward()
        optimizer.step()
        model.constraint()



