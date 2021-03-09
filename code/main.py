import numpy as np
import scipy.sparse as sp
import random
import gc

from clac_metric import get_metrics
from utils import constructHNet, constructNet, get_edge_index, Sizes
import torch as t
from torch import optim
from loss import Myloss

import MKGCN


def train(model, train_data, optimizer, sizes):
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(train_data['Y_train'], score, model.drug_l, model.mic_l, model.alpha1,
                               model.alpha2, sizes)
        model.alpha1 = t.mm(
            t.mm((t.mm(model.drug_k, model.drug_k) + model.lambda1 * model.drug_l).inverse(), model.drug_k),
            2 * train_data['Y_train'] - t.mm(model.alpha2.T, model.mic_k.T)).detach()
        model.alpha2 = t.mm(t.mm((t.mm(model.mic_k, model.mic_k) + model.lambda2 * model.mic_l).inverse(), model.mic_k),
                            2 * train_data['Y_train'].T - t.mm(model.alpha1.T, model.drug_k.T)).detach()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(1, sizes.epoch + 1):
        train_reg_loss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass


def PredictScore(train_drug_mic_matrix, drug_matrix, mic_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['Y_train'] = t.DoubleTensor(train_drug_mic_matrix)
    Heter_adj = constructHNet(train_drug_mic_matrix, drug_matrix, mic_matrix)
    Heter_adj = t.FloatTensor(Heter_adj)
    Heter_adj_edge_index = get_edge_index(Heter_adj)
    train_data['Adj'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}

    X = constructNet(train_drug_mic_matrix)
    X = t.FloatTensor(X)
    train_data['feature'] = X

    model = MKGCN.Model(sizes, drug_matrix, mic_matrix)
    print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())

    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate)

    train(model, train_data, optimizer, sizes)
    return model(train_data)


def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)
    random.shuffle(random_index)
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index


def cross_validation_experiment(drug_mic_matrix, drug_matrix, mic_matrix, sizes):
    index = crossval_index(drug_mic_matrix, sizes)
    metric = np.zeros((1, 7))
    pre_matrix = np.zeros(drug_mic_matrix.shape)
    print("seed=%d, evaluating drug-microbe...." % (sizes.seed))
    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_mic_matrix, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0
        drug_len = drug_mic_matrix.shape[0]
        dis_len = drug_mic_matrix.shape[1]
        drug_mic_res = PredictScore(
            train_matrix, drug_matrix, mic_matrix, sizes.seed, sizes)
        predict_y_proba = drug_mic_res.reshape(drug_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
        metric_tmp = get_metrics(drug_mic_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / sizes.k_fold)
    metric = np.array(metric / sizes.k_fold)
    return metric, pre_matrix


if __name__ == "__main__":
    data_path = '../Data/MicrobeDrugA/'
    # data_set = 'aBiofilm/'
    data_set = 'MDAD/'
    # data_set = 'DrugVirus/'
    drug_sim = np.loadtxt(data_path + data_set + 'drugsimilarity.txt', delimiter='\t')
    mic_sim = np.loadtxt(data_path + data_set + 'microbesimilarity.txt', delimiter='\t')
    adj_triple = np.loadtxt(data_path + data_set + 'adj.txt')
    drug_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(drug_sim), len(mic_sim))).toarray()

    average_result = np.zeros((1, 7), float)
    circle_time = 1
    sizes = Sizes(drug_sim.shape[0], mic_sim.shape[0])
    results = []

    result, pre_matrix = cross_validation_experiment(
        drug_mic_matrix, drug_sim, mic_sim, sizes)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
