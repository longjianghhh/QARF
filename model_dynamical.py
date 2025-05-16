import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
# from layers import GraphConvolution

class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        # self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[1], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        # enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h2))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[1])
        # self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        # dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h1))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar



class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()
        # dims0 = []
        # for idim in range(n_stacks-2):
        #     linshidim=round(n_input[0]*0.8)
        #     linshidim = int(linshidim)
        #     dims0.append(linshidim)
        # linshidim = 1500
        # linshidim = int(linshidim)
        # dims0.append(linshidim)

        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 3):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        # encoder0
        # self.enc0_1 = Linear(n_input[0], dims0[0])
        # self.enc0_2 = Linear(dims0[0], dims0[1])
        # self.enc0_3 = Linear(dims0[1], dims0[2])
        # self.z0_layer = Linear(dims0[2], n_z)
        # self.z0_b0 = nn.BatchNorm1d(n_z)

        # # decoder0
        # self.dec0_0 = Linear(n_z, n_z)
        # self.dec0_1 = Linear(n_z, dims0[2])
        # self.dec0_2 = Linear(dims0[2], dims0[1])
        # self.dec0_3 = Linear(dims0[1], dims0[0])
        # self.x0_bar_layer = Linear(dims0[0], n_input[0])
 
        self.regression = Linear(n_z, nLabel)
        self.act = nn.Sigmoid()
        
        self.regression0 = Linear(n_z, nLabel)
        self.regression1 = Linear(n_z, nLabel)
        self.regression2 = Linear(n_z, nLabel)
        self.regression3 = Linear(n_z, nLabel)
        self.regression4 = Linear(n_z, nLabel)
        self.regression5 = Linear(n_z, nLabel)
        self.view_classifier=[self.regression0,self.regression1,self.regression2,self.regression3,self.regression4,self.regression5]
        # self.act = nn.Sigmoid()
        
        self.variable0 = nn.Parameter(torch.Tensor([1/6]))
        self.variable1 = nn.Parameter(torch.Tensor([1/6]))
        self.variable2 = nn.Parameter(torch.Tensor([1/6]))
        self.variable3 = nn.Parameter(torch.Tensor([1/6]))
        self.variable4 = nn.Parameter(torch.Tensor([1/6]))
        self.variable5 = nn.Parameter(torch.Tensor([1/6]))
                
        

    def forward(self, mul_X, we):

       
        summ = 0
        individual_zs = []
        loss_pro=0
        sim1=[]
        for enc_i, enc in enumerate(self.encoder_list):
            x = F.normalize(mul_X[enc_i])
            sim = torch.matmul(x,x.T)
            sim1.append(sim)
            z_i = enc(mul_X[enc_i])                      
            individual_zs.append(z_i)
    
        # #结合掩码归一化
        we[:,0]*=self.variable0
        we[:,1]*=self.variable1
        we[:,2]*=self.variable2
        we[:,3]*=self.variable3
        we[:,4]*=self.variable4
        we[:,5]*=self.variable5
        
        row_sum=we.sum(dim=1)
        norm_we=we/row_sum.unsqueeze(1)
        z = torch.diag(norm_we[:,0]).mm(individual_zs[0])+torch.diag(norm_we[:,1]).mm(individual_zs[1])+torch.diag(norm_we[:,2]).mm(individual_zs[2])+torch.diag(norm_we[:,3]).mm(individual_zs[3])\
        +torch.diag(norm_we[:,4]).mm(individual_zs[4])+torch.diag(norm_we[:,5]).mm(individual_zs[5])
        
        z_norm = F.normalize(z)
        sim2 = torch.matmul(z_norm,z_norm.T) 
        
        for i in range(len(mul_X)):
            ret = (torch.diag(we[:,i]).mm(sim1[i] - sim2).mm(torch.diag(we[:,i]))) ** 2
            ret = torch.mean(ret)
            loss_pro+=ret

        x_bar_list = []
        y_specific=[]  
        
        for dec_i, dec in enumerate(self.decoder_list):          
            y_specific.append(self.act(self.regression(F.relu(individual_zs[dec_i]))))
           
        
        yLable = self.act(self.regression(F.relu(z)))
        return x_bar_list, yLable, z, individual_zs, y_specific,loss_pro/6


class DICNet(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DICNet, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we):
        x_bar_list, target_pre, fusion_z, individual_zs,y_specific ,loss_pro= self.ae(mul_X, we)

        return x_bar_list, target_pre, fusion_z, individual_zs,y_specific,loss_pro