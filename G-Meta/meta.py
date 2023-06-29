import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Classifier
from    copy import deepcopy

# 这段代码定义了一个名为euclidean_dist的函数，用于计算两个矩阵x和y之间的欧几里得距离。
# 其中，x和y的维度分别为N x D和M x D，其中N和M分别表示矩阵的行数，D表示矩阵的列数。
# 如果x和y的列数不相等，则会引发一个异常。
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    # 通过unsqueeze和expand操作将x和y扩展为三维张量，然后计算它们之间的差的平方，并在最后一个维度上求和，
    y = y.unsqueeze(0).expand(n, m, d)

    # 得到一个N x M的矩阵，表示x中每个向量与y中每个向量之间的欧几里得距离的平方 
    return torch.pow(x - y, 2).sum(2)

# 计算支持集上的原型损失
# logits是一个张量，表示模型的输出
# y_t是一个张量，表示目标标签，
# n_support是一个整数，表示每个类别的支持集大小
def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')
    # 首先将y_t和logits转换为CPU上的张量，然后定义了一个名为supp_idxs的函数，用于获取每个类别的支持集索引。
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # 通过torch.unique函数获取所有类别，并计算类别数和查询数
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = n_support

    # 对于每个类别，获取其支持集索引，并计算其原型。
    support_idxs = list(map(supp_idxs, classes))
    # prototype是支持集子图嵌入的平均值
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # 获取每个类别的查询集索引，并计算查询样本与原型之间的欧几里得距离
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]   
    dists = euclidean_dist(query_samples, prototypes)

    # 通过log_softmax函数计算损失值和准确率，并返回损失值、准确率和原型 
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    # target_inds是一个张量，用于存储每个查询样本的目标标签
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes

# 计算查询集上的原型损失
def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


import torch.nn.functional as F

def proto_loss_qry_syn(logits, y_t, prototypes, reg_lambda=0.01):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0] / n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    # Add L2 regularization term to the loss
    l2_reg = torch.norm(prototypes, p=2)
    loss_val += reg_lambda * l2_reg

    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val

class Meta(nn.Module):
    def __init__(self, args, config,device):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr # 内部循环的学习率
        self.meta_lr = args.meta_lr # 元学习的学习率
        self.n_way = args.n_way # 每个任务的类别数
        self.k_spt = args.k_spt # 每个类别的支持集大小
        self.k_qry = args.k_qry # 每个类别的查询集大小
        self.task_num = args.task_num # 任务数
        self.update_step = args.update_step # 内部循环的步数
        self.update_step_test = args.update_step_test # 测试时内部循环的步数
        self.device = device
        self.net = Classifier(config)
        self.net = self.net.to(self.device)
        
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.method = args.method

    def grad_opt_syn(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """

        task_num = len(x_spt)
        querysz = len(y_qry[0])
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        losses_opt= [0 for _ in range(task_num)]
        # print(task_num)
        # print(self.update_step + 1)
        # print(losses_opt)
        for i in range(task_num): # 遍历每个任务，并在支持集上运行内部循环，以更新分类器的参数
            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(self.device)
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(self.device)
            # 1. run the i-th task and compute loss for k=0
            logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, vars=None)
            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)

            # 首先使用支持集的数据和分类器的参数计算损失值，并计算梯度和快速权重
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, self.net.parameters())
                loss_q, acc_q = proto_loss_qry_syn(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # 使用快速权重在查询集上计算损失值和准确率，并将它们添加到损失值和准确率列表中
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                loss_q, acc_q = proto_loss_qry_syn(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            # 在内部循环的每个步骤中，使用快速权重计算损失值、梯度和快速权重，并使用它们来更新快速权重。
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry_syn(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

            # losses_opt[i]=losses_q[-1] 这一行直接执行的话会导致losses_opt[i]的值为一个变量，而不是一个数值，进而导致其无法进行反向传播

            losses_opt[i] =losses_q[-1].clone()
            # print(losses_opt)
            # if i >0:
            #     print(losses_opt[i],losses_opt[i-1])
            #     print(losses_opt[i] == losses_opt[i-1])
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        accs = np.array(corrects) / (task_num)
        if torch.isnan(loss_q):
            pass
        else:    
            # optimize theta parameters
            return losses_opt,loss_q
        
    def grad_opt(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """

        task_num = len(x_spt)
        querysz = len(y_qry[0])
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        losses_opt= [0 for _ in range(task_num)]
        # print(task_num)
        # print(self.update_step + 1)
        # print(losses_opt)
        for i in range(task_num): # 遍历每个任务，并在支持集上运行内部循环，以更新分类器的参数
            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(self.device)
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(self.device)
            # 1. run the i-th task and compute loss for k=0
            logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, vars=None)
            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)

            # 首先使用支持集的数据和分类器的参数计算损失值，并计算梯度和快速权重
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, self.net.parameters())
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # 使用快速权重在查询集上计算损失值和准确率，并将它们添加到损失值和准确率列表中
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            # 在内部循环的每个步骤中，使用快速权重计算损失值、梯度和快速权重，并使用它们来更新快速权重。
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

            # losses_opt[i]=losses_q[-1] 这一行直接执行的话会导致losses_opt[i]的值为一个变量，而不是一个数值，进而导致其无法进行反向传播

            losses_opt[i] =losses_q[-1].clone()
            # print(losses_opt)
            # if i >0:
            #     print(losses_opt[i],losses_opt[i-1])
            #     print(losses_opt[i] == losses_opt[i-1])
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        accs = np.array(corrects) / (task_num)
        if torch.isnan(loss_q):
            pass
        else:    
            # optimize theta parameters
            return losses_opt,loss_q
        #     self.meta_optim.zero_grad()
        #     loss_q.backward()
        #     # # 注册回调函数保存梯度
        #     # self.net.parameters().register_hook(save_gradients)
        #     self.meta_optim.step()

        # # 最后，它计算所有任务的平均准确率，并使用反向传播算法更新分类器的参数 
        # accs = np.array(corrects) / (task_num)

        # return accs,self.net.parameters()
    # 在支持集和查询集上训练和测试一个分类器
    # x_spt是一个大小为b的列表，其中每个元素是一个支持集的小批量子图，
    # y_spt是一个大小为b x setsz的张量，其中b表示任务数，setsz表示每个任务的支持集大小，
    # x_qry是一个大小为b的列表，其中每个元素是一个查询集的小批量子图，
    # y_qry是一个大小为b x querysz的张量，其中querysz表示每个任务的查询集大小，
    # c_spt和c_qry是一个大小为b x setsz和b x querysz的张量，用于存储每个样本的类别标签，
    # n_spt和n_qry是一个大小为b x setsz和b x querysz的张量，用于存储每个样本的节点索引，
    # g_spt和g_qry是一个大小为b x setsz和b x querysz的张量，用于存储每个样本所属的子图索引，
    # feat是一个大小为n_graphs x n_nodes x n_features的张量，用于存储每个子图中每个节点的特征向量。
    def forward_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_spt)
        querysz = len(y_qry[0])
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num): # 遍历每个任务，并在支持集上运行内部循环，以更新分类器的参数
            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(self.device)
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(self.device)
            # 1. run the i-th task and compute loss for k=0
            logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, vars=None)
            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)

            # 首先使用支持集的数据和分类器的参数计算损失值，并计算梯度和快速权重
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, self.net.parameters())
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # 使用快速权重在查询集上计算损失值和准确率，并将它们添加到损失值和准确率列表中
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            # 在内部循环的每个步骤中，使用快速权重计算损失值、梯度和快速权重，并使用它们来更新快速权重。
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(x_spt[i].to(self.device), c_spt[i].to(self.device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(self.device), c_qry[i].to(self.device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        
        if torch.isnan(loss_q):
            pass
        else:    
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            # # 注册回调函数保存梯度
            # self.net.parameters().register_hook(save_gradients)
            self.meta_optim.step()

        # 最后，它计算所有任务的平均准确率，并使用反向传播算法更新分类器的参数 
        accs = np.array(corrects) / (task_num)

        return accs,self.net.parameters()
    
    
    # 只处理一个任务，更新步数由 self.update_step_test 控制
    def finetunning_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        querysz = len(y_qry[0])

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # finetunning on the copied model instead of self.net
        net = deepcopy(self.net) # 创建一个深度拷贝的模型 net = deepcopy(self.net)，然后在这个拷贝的模型上进行微调
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        n_spt = n_spt[0]
        n_qry = n_qry[0]
        g_spt = g_spt[0]
        g_qry = g_qry[0]

        feat_spt = torch.Tensor(np.vstack(([feat[g_spt[j]][np.array(x)] for j, x in enumerate(n_spt)]))).to(self.device)
        feat_qry = torch.Tensor(np.vstack(([feat[g_qry[j]][np.array(x)] for j, x in enumerate(n_qry)]))).to(self.device)
            

        # 1. run the i-th task and compute loss for k=0
        logits, _ = net(x_spt.to(self.device), c_spt.to(self.device), feat_spt)
        loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(self.device), c_qry.to(self.device), feat_qry, net.parameters())
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[0] = corrects[0] + acc_q
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(self.device), c_qry.to(self.device), feat_qry, fast_weights)
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[1] = corrects[1] + acc_q


        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits, _ = net(x_spt.to(self.device), c_spt.to(self.device), feat_spt, fast_weights)
            loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q, _ = net(x_qry.to(self.device), c_qry.to(self.device), feat_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[k + 1] = corrects[k + 1] + acc_q

        del net
        accs = np.array(corrects) 

        return accs

    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        if self.method == 'G-Meta':
            accs = self.forward_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        if self.method == 'G-Meta':
            accs = self.finetunning_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        return accs
