import torch as t
from torch import nn
from torch_geometric.nn import conv
from utils import *


class Model(nn.Module):
    def __init__(self, sizes, drug_sim, mic_sim):
        super(Model, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.drug_size = sizes.drug_size
        self.mic_size = sizes.mic_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3
        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma

        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 4
        self.drug_ps = t.ones(self.kernel_len) / self.kernel_len
        self.mic_ps = t.ones(self.kernel_len) / self.kernel_len

        self.drug_sim = t.DoubleTensor(drug_sim)
        self.mic_sim = t.DoubleTensor(mic_sim)

        self.gcn_1 = conv.GCNConv(self.drug_size + self.mic_size, self.F1)
        self.gcn_2 = conv.GCNConv(self.F1, self.F2)
        self.gcn_3 = conv.GCNConv(self.F2, self.F3)

        self.alpha1 = t.randn(self.drug_size, self.mic_size).double()
        self.alpha2 = t.randn(self.mic_size, self.drug_size).double()

        self.drug_l = []
        self.mic_l = []

        self.drug_k = []
        self.mic_k = []

    def forward(self, input):
        t.manual_seed(self.seed)
        x = input['feature']
        adj = input['Adj']
        drugs_kernels = []
        mic_kernels = []
        H1 = t.relu(self.gcn_1(x, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H1[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))

        H2 = t.relu(self.gcn_2(H1, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.drug_size].clone(), 0, self.h2_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H2[self.drug_size:].clone(), 0, self.h2_gamma, True).double()))

        H3 = t.relu(self.gcn_3(H2, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.drug_size].clone(), 0, self.h3_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H3[self.drug_size:].clone(), 0, self.h3_gamma, True).double()))

        drugs_kernels.append(self.drug_sim)
        mic_kernels.append(self.mic_sim)

        drug_k = sum([self.drug_ps[i] * drugs_kernels[i] for i in range(len(self.drug_ps))])
        self.drug_k = normalized_kernel(drug_k)
        mic_k = sum([self.mic_ps[i] * mic_kernels[i] for i in range(len(self.mic_ps))])
        self.mic_k = normalized_kernel(mic_k)
        self.drug_l = laplacian(drug_k)
        self.mic_l = laplacian(mic_k)

        out1 = t.mm(self.drug_k, self.alpha1)
        out2 = t.mm(self.mic_k, self.alpha2)

        out = (out1 + out2.T) / 2

        return out
