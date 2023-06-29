import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import csv
import random
import pickle
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import itertools

# - root：数据集的根目录
# - mode：数据集的模式（如训练、验证等）
# - subgraph2label：子图到标签的映射
# - n_way：n-way分类问题
# - k_shot：支持集中每个类别的样本数
# - k_query：查询集中每个类别的样本数
# - batchsz：批次大小
# - args：其他参数
# - adjs：邻接矩阵列表
# - h：跳数
class Subgraphs(Dataset):
    def __init__(self, root, mode, subgraph2label, n_way, k_shot, k_query, batchsz, args, adjs, h):
        self.batchsz = batchsz  # batch of set, not batch of subgraphs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.k_query = k_query  # for query set
        self.setsz = self.n_way * self.k_shot  # num of samples per support set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.h = h # number of h hops
        self.sample_nodes = args.sample_nodes
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, %d-hops' % (
        mode, batchsz, n_way, k_shot, k_query, h))
    
        # load subgraph list if preprocessed
        self.subgraph2label = subgraph2label
        
        self.link_pred_mode = False
        

        dictLabels = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        # print("dictLabels:",dictLabels) # 某个文件中节点-label的映射eg:dictLabels: {'6': ['0_6',
        # print("dictGraphs:",dictGraphs) # 某个文件中一个图对应了那些节点的映射eg:dictGraphs: {0: ['0_6', '0_14',
        # print("dictGraphsLabels",dictGraphsLabels) # 某个文件中包含了哪些图以及图中的节点-label的映射eg:dictGraphsLabels {0: {'6': ['0_6',
    
        self.task_setup = args.task_setup

        self.G = []

        for i in adjs:
            self.G.append(i)
        
        self.subgraphs = {}
       
        if self.task_setup == 'Disjoint':
            self.data = []

            for i, (k, v) in enumerate(dictLabels.items()):
                self.data.append(v)  # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
            self.cls_num = len(self.data)

            self.create_batch_disjoint(self.batchsz)



    def loadCSV(self, csvf):

        dictLabels = {}

        # dictLabels（标签到子图的映射）、

        with open(csvf) as csvfile:
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

    def create_batch_disjoint(self, batchsz): # 为不相交标签设置创建任务批次。
        """
        create the entire set of batches of tasks for disjoint label setting, indepedent of # of graphs.
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #print(self.cls_num)
            #print(self.n_way)
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                
                # 2. select k_shot + k_query for each class
                
                selected_subgraphs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)

                np.random.shuffle(selected_subgraphs_idx)
                indexDtrain = np.array(selected_subgraphs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_subgraphs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * n_way)] numbers of subgraphs   
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    
    # helper to generate subgraphs on the fly.
    # 根据给定的图G、节点i和子图标识符item生成子图
    def generate_subgraph(self, G, i, item):
        if item in self.subgraphs:
            return self.subgraphs[item] # 检查item是否已经在self.subgraphs字典中。如果已经存在，直接返回对应的子图
        else:
            # instead of calculating shortest distance, we find the following ways to get subgraphs are quicker
            # 据self.h的值（跳数）来确定如何生成子图
            if self.h == 2:
                # 计算节点i的一阶邻居和二阶邻居（两跳邻居），并将它们与节点i一起存储在h_hops_neighbor中
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
            elif self.h == 1:
                # 计算节点i的一阶邻居（一跳邻居），并将它们与节点i一起存储在h_hops_neighbor中
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [i]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
            
            # 检查h_hops_neighbor中的节点数量是否大于self.sample_nodes。
            # 如果大于，函数会从h_hops_neighbor中随机选择self.sample_nodes个节点，并确保节点i也包含在选择的节点中
            if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace = False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))
            
            sub = G.subgraph(h_hops_neighbor) 

            # 获取子图sub中的节点ID（h_c），并创建一个字典dict_，将原始节点ID映射到子图中的新节点ID        
            h_c = list(sub.ndata[dgl.NID].numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.subgraphs[item] = (sub, dict_[i], h_c)
            
            return sub, dict_[i], h_c
    
    
    def __getitem__(self, index): # 根据给定的索引值（index），从数据集中获取一个任务，包括支持集（support set）和查询集（query set）
        """
        get one task. support_x_batch[index], query_x_batch[index]

        """
        #print(self.support_x_batch[index])
        # 生成子图信息
        
        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.support_x_batch[index] for item in sublist]
        
        # 从支持集（self.support_x_batch[index]）和查询集（self.query_x_batch[index]）中提取子图信息，
        # 包括子图索引、子图标签、中心节点和节点索引。这些信息将用于后续的图神经网络模型训练
        support_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]
        
        support_x = [i for i, j, k in info]
        support_y = np.array([self.subgraph2label[item]  
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        
        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        support_node_idx = [k for i, j, k in info]

        
        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.query_x_batch[index] for item in sublist]

        query_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.query_x_batch[index] for item in sublist]
        
        query_x = [i for i, j, k in info]
        query_y = np.array([self.subgraph2label[item]  
                              for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        
        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        if self.task_setup == 'Disjoint':
            unique = np.unique(support_y)
            random.shuffle(unique)
            # relative means the label ranges from 0 to n-way
            support_y_relative = np.zeros(self.setsz)
            query_y_relative = np.zeros(self.querysz)
            for idx, l in enumerate(unique):
                support_y_relative[support_y == l] = idx
                query_y_relative[query_y == l] = idx
            # this is a set of subgraphs for one task.
            # 使用 dgl.batch 方法将支持集和查询集的子图批量化，以便在图神经网络中进行批量处理
            batched_graph_spt = dgl.batch(support_x)
            batched_graph_qry = dgl.batch(query_x)

            return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(query_y_relative), torch.LongTensor(support_center), torch.LongTensor(query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx


class Subgraphs_syn(Dataset):
    def __init__(self, mode, subgraph2label, n_way, k_shot, k_query, batchsz, args, adjs, h,labels):
        self.batchsz = batchsz  # batch of set, not batch of subgraphs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.k_query = k_query  # for query set
        self.setsz = self.n_way * self.k_shot  # num of samples per support set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.h = h # number of h hops
        self.sample_nodes = args.sample_nodes
        # print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, %d-hops' % (
        # mode, batchsz, n_way, k_shot, k_query, h))
    
        # load subgraph list if preprocessed
        self.subgraph2label = subgraph2label
        
        self.link_pred_mode = False
        
        dictLabels = self.loadCSV(labels)  # csv path
        # print("dictLabels:",dictLabels) # 某个文件中节点-label的映射eg:dictLabels: {'6': ['0_6',
        # print("dictGraphs:",dictGraphs) # 某个文件中一个图对应了那些节点的映射eg:dictGraphs: {0: ['0_6', '0_14',
        # print("dictGraphsLabels",dictGraphsLabels) # 某个文件中包含了哪些图以及图中的节点-label的映射eg:dictGraphsLabels {0: {'6': ['0_6',
    
        self.task_setup = args.task_setup

        self.G = []
        self.G.append(adjs)

 
        
        self.subgraphs = {}
       
        if self.task_setup == 'Disjoint':
            self.data = []

            for i, (k, v) in enumerate(dictLabels.items()):
                self.data.append(v)  # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
            self.cls_num = len(self.data)

            self.create_batch_disjoint(self.batchsz)


    def loadCSV(self, labels):

        label_dict = {}
        for i, label in enumerate(labels):
            node_id = f"0_{i}"
            label = str(label.item())
            if label in label_dict:
                label_dict[label].append(node_id)
            else:
                label_dict[label] = [node_id]
        return label_dict


    def create_batch_disjoint(self, batchsz): # 为不相交标签设置创建任务批次。
        """
        create the entire set of batches of tasks for disjoint label setting, indepedent of # of graphs.
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #print(self.cls_num)
            #print(self.n_way)
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                
                # 2. select k_shot + k_query for each class

                if len(self.data[cls]) < self.k_shot + self.k_query:
                    selected_subgraphs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, True)
                else:
                    selected_subgraphs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)

                np.random.shuffle(selected_subgraphs_idx)
                indexDtrain = np.array(selected_subgraphs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_subgraphs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * n_way)] numbers of subgraphs   
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    
    # helper to generate subgraphs on the fly.
    # 根据给定的图G、节点i和子图标识符item生成子图
    def generate_subgraph(self, G, i, item):
        if item in self.subgraphs:
            return self.subgraphs[item] # 检查item是否已经在self.subgraphs字典中。如果已经存在，直接返回对应的子图
        else:
            # instead of calculating shortest distance, we find the following ways to get subgraphs are quicker
            # 据self.h的值（跳数）来确定如何生成子图
            if self.h == 2:
                # 计算节点i的一阶邻居和二阶邻居（两跳邻居），并将它们与节点i一起存储在h_hops_neighbor中
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
            elif self.h == 1:
                # 计算节点i的一阶邻居（一跳邻居），并将它们与节点i一起存储在h_hops_neighbor中
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [i]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
            
            # 检查h_hops_neighbor中的节点数量是否大于self.sample_nodes。
            # 如果大于，函数会从h_hops_neighbor中随机选择self.sample_nodes个节点，并确保节点i也包含在选择的节点中
            if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace = False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))
            
            sub = G.subgraph(h_hops_neighbor) 

            # 获取子图sub中的节点ID（h_c），并创建一个字典dict_，将原始节点ID映射到子图中的新节点ID        
            h_c = list(sub.ndata[dgl.NID].numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.subgraphs[item] = (sub, dict_[i], h_c)
          
            return sub, dict_[i], h_c
    
    
    def __getitem__(self, index): # 根据给定的索引值（index），从数据集中获取一个任务，包括支持集（support set）和查询集（query set）
        """
        get one task. support_x_batch[index], query_x_batch[index]

        """
        #print(self.support_x_batch[index])
        # 生成子图信息

        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.support_x_batch[index] for item in sublist]

        # 从支持集（self.support_x_batch[index]）和查询集（self.query_x_batch[index]）中提取子图信息，
        # 包括子图索引、子图标签、中心节点和节点索引。这些信息将用于后续的图神经网络模型训练
        support_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]
        
        support_x = [i for i, j, k in info]
        
        support_y = np.array([self.subgraph2label[item]  
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        
        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        support_node_idx = [k for i, j, k in info]

        
        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.query_x_batch[index] for item in sublist]
        
        query_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.query_x_batch[index] for item in sublist]
        
        query_x = [i for i, j, k in info]
        query_y = np.array([self.subgraph2label[item]  
                              for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        
        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        if self.task_setup == 'Disjoint':
            unique = np.unique(support_y)
            random.shuffle(unique)
            # relative means the label ranges from 0 to n-way
            support_y_relative = np.zeros(self.setsz)
            query_y_relative = np.zeros(self.querysz)
            for idx, l in enumerate(unique):
                support_y_relative[support_y == l] = idx
                query_y_relative[query_y == l] = idx
            # this is a set of subgraphs for one task.
            # 使用 dgl.batch 方法将支持集和查询集的子图批量化，以便在图神经网络中进行批量处理
            batched_graph_spt = dgl.batch(support_x)
            batched_graph_qry = dgl.batch(query_x)

            return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(query_y_relative), torch.LongTensor(support_center), torch.LongTensor(query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx


    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


