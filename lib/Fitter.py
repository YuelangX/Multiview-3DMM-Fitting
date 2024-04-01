import torch
import numpy as np
from einops import rearrange


class Fitter():
    def __init__(self, cfg, dataset, face_model, camera, recorder, device):
        self.cfg = cfg
        self.dataset = dataset
        self.face_model = face_model
        self.camera = camera
        self.recorder = recorder
        self.device = device

        self.optimizers = [torch.optim.Adam([{'params' : self.face_model.scale, 'lr' : 1e-3},
                                             {'params' : self.face_model.pose, 'lr' : 1e-2}]),
                           torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]
    
    def run(self):
        landmarks_gt, extrinsics0, intrinsics0, frames = self.dataset.get_item()
        landmarks_gt = torch.from_numpy(landmarks_gt).float().to(self.device)
        extrinsics0 = torch.from_numpy(extrinsics0).float().to(self.device)
        intrinsics0 = torch.from_numpy(intrinsics0).float().to(self.device)
        extrinsics = rearrange(extrinsics0, 'b v x y -> (b v) x y')
        intrinsics = rearrange(intrinsics0, 'b v x y -> (b v) x y')
        
        for optimizer in self.optimizers:
            pprev_loss = 1e8
            prev_loss = 1e8

            for i in range(int(1e10)):
                _, landmarks_3d = self.face_model()
                landmarks_3d = landmarks_3d.unsqueeze(1).repeat(1, landmarks_gt.shape[1], 1, 1)
                landmarks_3d = rearrange(landmarks_3d, 'b v x y -> (b v) x y')

                landmarks_2d = self.project(landmarks_3d, intrinsics, extrinsics)
                landmarks_2d = rearrange(landmarks_2d, '(b v) x y -> b v x y', b=landmarks_gt.shape[0])

                pro_loss = (((landmarks_2d / self.cfg.image_size - landmarks_gt[:, :, :, 0:2] / self.cfg.image_size) * landmarks_gt[:, :, :, 2:3]) ** 2).sum(-1).sum(-2).mean()
                reg_loss = self.face_model.reg_loss(self.cfg.reg_id_weight, self.cfg.reg_exp_weight)
                loss = pro_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if abs(loss.item() - prev_loss) < 1e-10 and abs(loss.item() - pprev_loss) < 1e-9:
                    break
                else:
                    pprev_loss = prev_loss
                    prev_loss = loss.item()

        log = {
            'frames': frames,
            'landmarks_gt': landmarks_gt,
            'landmarks_2d': landmarks_2d.detach(),
            'face_model': self.face_model,
            'intrinsics': intrinsics0,
            'extrinsics': extrinsics0
        }
        self.recorder.log(log)


    def project(self, points_3d, intrinsic, extrinsic):
        points_3d = points_3d.permute(0,2,1)
        calibrations = torch.bmm(intrinsic, extrinsic)
        points_2d = self.camera.project(points_3d, calibrations)
        points_2d = points_2d.permute(0,2,1)
        return points_2d