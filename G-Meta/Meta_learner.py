from  subgraph_data_processing import Subgraphs_syn
import torch
from torch.utils.data import DataLoader
import numpy as np
import dgl
from utils import *
import deeprobust.graph.utils as utils
import torch.nn as nn
from utils import match_loss, regularization, row_normalize_tensor
from models.parametrized_adj import PGE
from meta import Meta
import copy

def collate(samples):
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

    return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx

class Meta_learner():
    # 初始化函数，接收数据、参数和设备信息
    def __init__(self, db_train,db_val,db_test, args, device='cuda', feat_shape= None,**kwargs):
        self.db_train = db_train
        self.db_val = db_val
        self.db_test = db_test
        self.args = args
        self.device = device
        
        path = r"../DATA/arxiv/train.csv"
        
        ratio = 0.0025 # 0.25%
        n,labels_syn= self.generate_labels_from_csv(path,ratio)

        # 计算需要生成的synthetic graph的节点数
        n = int(n)        
        # 初始化synthetic graph的节点数和特征
        # 获取特征维度
        d = 128
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        # 初始化PGE模型
        self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=self.args).to(device)
        
        # 生成synthetic graph的标签
        self.labels_syn = torch.LongTensor(labels_syn).to(device)

        config = [('GraphConv', [feat_shape, self.args.hidden_dim])]

        if self.args.h > 1:
            config = config + [('GraphConv', [self.args.hidden_dim, self.args.hidden_dim])] * (self.args.h - 1)

        config = config + [('Linear', [self.args.hidden_dim, args.n_way])]
        self.maml = Meta(args, config).to(device)
        # 重置synthetic graph的特征
        self.reset_parameters()

        self.dictLabels = self.loadCSV(path)
        # 初始化特征优化器和PGE模型优化器
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        
        # 打印synthetic graph的节点数和特征维度
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def loadCSV(self,path):
        import csv
        from collections import Counter
        dictLabels = {}

        # dictLabels（标签到子图的映射）、

        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                g_idx = int(filename.split('_')[0])
                label = row[2]
                # append filename to current label

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]

        return dictLabels

    def generate_labels_from_csv(self,file_path,ratio):
        import csv
        from collections import Counter
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            data = list(reader)

        counter = Counter()  # 使用 Counter 对标签进行计数
        num_class_dict = {}  # 存储每个类别需要生成的节点数
        n = len(data)  # 数据总数

        # 统计每个类别的出现次数
        for row in data:
            label = int(row['label'])
            counter[label] += 1

        # 根据标签出现次数进行排序
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []  # 合成标签列表
        syn_class_indices = {}  # 记录每个类别的索引范围
    #     ratio = 0.08
        # 计算每个类别需要生成的节点数，并生成合成标签
        for ix, (c, num) in enumerate(sorted_counter):

            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * ratio) - sum_  
                # 如果是最后一个类别，那么直接计算需要生成的节点数，并将这个类别的合成标签添加到 labels_syn 列表中
                syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

            else:
                num_class_dict[c] = max(int(num * ratio), 1)  # 其他类别的节点数
                # 如果不是最后一个类别，那么先计算需要生成的节点数，然后将这个类别的合成标签添加到 labels_syn 列表中。
                sum_ += num_class_dict[c]
                syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        return n * ratio,labels_syn

    def get_sub_adj_feat(self,features, labels_syn, label_dict):
        idx_selected = []

        from collections import Counter
        counter = Counter(labels_syn.cpu().numpy())
        selected_features = []
        for c in counter.keys():
            class_label = str(c)
            if class_label in label_dict:
                node_indices = [label_dict[class_label].index(node) for node in label_dict[class_label]]
                num_features = counter[c]
                selected_indices = np.random.choice(node_indices, size=num_features, replace=False)

                idx_selected.extend(selected_indices)

                selected_features.extend(features[selected_indices])

        selected_features = np.array(selected_features)

        return selected_features
    
    def generate_syn_feat_db(self,dictLabels,feat):

#         dictLabels = loadCSV(path)
        self.feat_syn = torch.from_numpy(self.get_sub_adj_feat(feat, self.labels_syn, dictLabels)).to(self.device)
    # # 重置synthetic graph的特征
    # reset_parameters()
        adj_syn_orj = self.pge(self.feat_syn)
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn_orj, sparse=False)

        adj_syn = adj_syn_norm
        adj_syn[adj_syn < 0.5] = 0
        adj_syn[adj_syn >= 0.5] = 1

        adj_syn_norm_sp = sp.csr_matrix(adj_syn.cpu().detach().numpy())
        g = dgl.from_scipy(adj_syn_norm_sp) # 生成的合成图太过密集，导致二跳子图约等于全图
        # 将labels_syn转换为dictLabels形式的字典
        dict_labels = {f"0_{i}": self.labels_syn[i].item() for i in range(len(self.labels_syn))}
        db_syn = Subgraphs_syn('syn', dict_labels, n_way=self.args.n_way, k_shot=self.args.k_spt,k_query=self.args.k_qry, batchsz=100, args = self.args, adjs = g, h = self.args.h, labels = self.labels_syn)
       

        db = DataLoader(db_syn, self.args.task_num, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, collate_fn = collate)
        
        return db,adj_syn_orj
    
    def train(self, verbose=True,feat = None,dgl_graph = None):
        # 获取参数和数据
        args = self.args
        # data = self.data

        db_train = self.db_train
        db_val = self.db_val
        db_test = self.db_test
        max_acc = 0
        model_parameters = list(self.maml.parameters())


        for epoch in range(self. args.epoch):
            db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
            db_syn,adj_syn  = self.generate_syn_feat_db(self.dictLabels,feat[0])
            loss = torch.tensor(0.0).to(self.device)
            for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db): # 处理每个任务
                nodes_len = 0
                # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
                # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.                
                nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])
                loss_q_real = self.maml.grad_opt(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
                gw_real = torch.autograd.grad(loss_q_real, model_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
            

            for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db_syn): # 处理每个任务
                nodes_len = 0
                # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
                # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.                
                nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])
                loss_q_syn = self.maml.grad_opt(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
                gw_syn  = torch.autograd.grad(loss_q_syn , model_parameters)
                gw_syn  = list((_.detach().clone() for _ in gw_syn ))

            # 计算匹配损失
            
            loss += match_loss(gw_syn, gw_real, args, device=self.device)

            # 计算正则化损失
            # TODO: regularize
            if args.alpha > 0:
                loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(self.labels_syn))
            # else:
            else:
                loss_reg = torch.tensor(0)

            loss = loss + loss_reg

            # 更新合成图
            self.optimizer_feat.zero_grad()
            # 每次计算梯度时，梯度都会被累加到梯度缓存中。
            # 因此，在每次更新模型参数之前需要将梯度缓存清零，以避免梯度累加的影响。
            self.optimizer_pge.zero_grad()
            loss.backward() # 计算损失函数对于模型参数的梯度

            # 根据 it 的值选择更新 self.optimizer_pge 或 self.optimizer_feat
            if epoch % 2 == 0:
                self.optimizer_pge.step() # 使用优化算法来更新模型参数
            else:
                self.optimizer_feat.step()

            if args.debug and epoch % 2 ==0: # 打印梯度匹配损失
                print('Gradient matching loss:', loss.item())

            # if ol == outer_loop - 1:
            #         # print('loss_reg:', loss_reg.item())
            #         # print('Gradient matching loss:', loss.item())
            #     break
                            # 进行内循环，更新 GNN 模型的参数
            self.maml.meta_optim.zero_grad()
            loss_q_syn.backward()
            # # 注册回调函数保存梯度
            # self.net.parameters().register_hook(save_gradients)
            self.meta_optim.step() # update gnn param

            # validation per epoch
            db_v = DataLoader(db_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
            accs_all_test = []

            for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_v:

                accs = self.maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
                accs_all_test.append(accs)

            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Epoch:', epoch + 1, ' Val acc:', str(accs[-1])[:5])
            if accs[-1] > max_acc:
                max_acc = accs[-1]
                model_max = copy.deepcopy(self.maml)
    
        db_t = DataLoader(db_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
            accs = self.maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
            accs_all_test.append(accs)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', str(accs[1])[:5])

    