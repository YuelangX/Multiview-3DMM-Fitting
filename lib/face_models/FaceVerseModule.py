import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import so3_exponential_map


class FaceVerseModule(nn.Module):
    def __init__(self, batch_size):
        super(FaceVerseModule, self).__init__()

        self.id_dims = 150
        self.exp_dims = 52
        self.batch_size = batch_size

        model_dict = np.load('assets/FaceVerse/faceverse_simple_v2.npy', allow_pickle=True).item()
        self.register_buffer('skinmask', torch.tensor(model_dict['skinmask']))
        kp_inds = torch.tensor(model_dict['keypoints']).squeeze().long()
        self.register_buffer('kp_inds', kp_inds)

        meanshape = torch.tensor(model_dict['meanshape'])
        meanshape[:, 1:] = -meanshape[:, 1:]
        self.register_buffer('meanshape', meanshape.view(1, -1).float())

        idBase = torch.tensor(model_dict['idBase']).view(-1, 3, self.id_dims).float()
        idBase[:, 1:, :] = -idBase[:, 1:, :]
        self.register_buffer('idBase', idBase.view(-1, self.id_dims))

        exBase = torch.tensor(model_dict['exBase']).view(-1, 3, self.exp_dims).float()
        exBase[:, 1:, :] = -exBase[:, 1:, :]
        self.register_buffer('exBase', exBase.view(-1, self.exp_dims))

        self.register_buffer('faces', torch.tensor(model_dict['tri']).long())

        self.id_coeff = nn.Parameter(torch.zeros(1, self.id_dims).float())
        self.exp_coeff = nn.Parameter(torch.zeros(self.batch_size, self.exp_dims).float())
        self.scale = nn.Parameter(torch.ones(1).float() * 0.3)
        self.pose = nn.Parameter(torch.zeros(self.batch_size, 6).float())

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def get_vs(self, id_coeff, exp_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.exBase, exp_coeff) + self.meanshape

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