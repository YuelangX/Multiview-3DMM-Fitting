import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from pytorch3d.transforms import so3_exponential_map


class BFMModule(nn.Module):
    def __init__(self, batch_size):
        super(BFMModule, self).__init__()

        self.id_dims = 80
        self.exp_dims = 64
        self.batch_size = batch_size

        model_dict = loadmat('assets/BFM/BFM09_model_info.mat')
        self.register_buffer('skinmask', torch.tensor(model_dict['skinmask']))
        kp_inds = torch.tensor(model_dict['keypoints']-1).squeeze().long()
        kp_inds = torch.cat([kp_inds[0:48], kp_inds[49:54], kp_inds[55:68]])
        self.register_buffer('kp_inds', kp_inds)
        self.register_buffer('meanshape', torch.tensor(model_dict['meanshape']).float())
        self.register_buffer('idBase', torch.tensor(model_dict['idBase']).float())
        self.register_buffer('expBase', torch.tensor(model_dict['exBase']).float())
        self.register_buffer('faces', torch.tensor(model_dict['tri']-1).long())

        
        self.id_coeff = nn.Parameter(torch.zeros(1, self.id_dims).float())
        self.exp_coeff = nn.Parameter(torch.zeros(self.batch_size, self.exp_dims).float())
        self.scale = nn.Parameter(torch.ones(1).float() * 0.15)
        self.pose = nn.Parameter(torch.zeros(self.batch_size, 6).float())

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def get_vs(self, id_coeff, exp_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - \
            self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def forward(self):
        id_coeff = self.id_coeff.repeat(self.batch_size, 1)
        vertices = self.get_vs(id_coeff, self.exp_coeff)
        R = so3_exponential_map(self.pose[:, :3])
        T = self.pose[:, 3:]
        vertices = torch.bmm(vertices * self.scale, R.permute(0,2,1)) + T[:, None, :]
        landmarks = self.get_lms(vertices)
        return vertices, landmarks

    def reg_loss(self, id_weight, exp_weight):
        id_reg_loss = (self.id_coeff ** 2).sum()
        exp_reg_loss = (self.exp_coeff ** 2).sum(-1).mean()
        return id_reg_loss * id_weight + exp_reg_loss * exp_weight

    def save(self, path, batch_id=-1):
        if batch_id < 0:
            id_coeff = self.id_coeff.detach().cpu().numpy()
            exp_coeff = self.exp_coeff.detach().cpu().numpy()
            scale = self.scale.detach().cpu().numpy()
            pose = self.pose.detach().cpu().numpy()
            np.savez(path, id_coeff=id_coeff, exp_coeff=exp_coeff, scale=scale, pose=pose)
        else:
            id_coeff = self.id_coeff.detach().cpu().numpy()
            exp_coeff = self.exp_coeff[batch_id:batch_id+1].detach().cpu().numpy()
            scale = self.scale.detach().cpu().numpy()
            pose = self.pose[batch_id:batch_id+1].detach().cpu().numpy()
            np.savez(path, id_coeff=id_coeff, exp_coeff=exp_coeff, scale=scale, pose=pose)