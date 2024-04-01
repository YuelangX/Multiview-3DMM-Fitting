import torch
import pyrender
import cv2
import numpy as np


class Camera():
    def __init__(self, image_size):
        self.image_size = image_size

        self.lights = []
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=60.0)
        for x in [-2, 2]:
            y = 2
            for z in [-2, 2]:
                light_pose = np.array([[1.0,  0.0,  0.0,   x],
                                        [0.0,  1.0,  0.0,   y],
                                        [0.0,  0.0,  1.0,   z],
                                        [0.0,  0.0,  0.0,   1.0]])
                self.lights.append((light, light_pose))

    def project(self, query_pts, calibrations):
        query_pts = torch.bmm(calibrations[:, :3, :3], query_pts)
        query_pts = query_pts + calibrations[:, :3, 3:4]
        query_pts_xy = query_pts[:, :2, :] / query_pts[:, 2:, :]
        query_pts_xy = query_pts_xy
        return query_pts_xy

    def init_renderer(self, intrinsic, extrinsic):
        self.R = extrinsic[0:3, 0:3]
        self.T = extrinsic[0:3, 3:4]
        self.K = intrinsic

        Rotate_y_180 = torch.eye(3).to(self.R.device)
        Rotate_y_180[0,0] = -1.0
        Rotate_y_180[2,2] = -1.0
        R_pyrender = torch.matmul(torch.inverse(self.R), Rotate_y_180).float()
        T_pyrender = -torch.matmul(torch.inverse(self.R), self.T)[:,0].float()

        self.renderer = pyrender.IntrinsicsCamera(self.K[0,0], self.K[1,1], self.image_size - self.K[0,2], self.image_size - self.K[1,2])
        self.camera_pose = np.eye(4)
        self.camera_pose[0:3,0:3] = R_pyrender.cpu().numpy()
        self.camera_pose[0:3,3] = T_pyrender.cpu().numpy()
    
    def render(self, mesh, return_mask=False):
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0.0, 0.0, 0.0])
        scene.add(mesh)
        for light in self.lights:
            scene.add(light[0], pose=light[1])
        scene.add(self.renderer, pose=self.camera_pose)
        osr = pyrender.OffscreenRenderer(self.image_size, self.image_size)
        color, depth = osr.render(scene)
        color = cv2.flip(color, -1)
        depth = cv2.flip(depth, -1)
        if return_mask:
            return color, (depth > 0).astype(np.uint8) * 255
        else:
            return color