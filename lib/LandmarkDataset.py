import torch
import numpy as np
import glob
import os
import random
import cv2
from skimage import io

class LandmarkDataset():

    def __init__(self, landmark_folder, camera_folder):

        self.frames = sorted(os.listdir(landmark_folder))
        self.landmark_folder = landmark_folder
        self.camera_folder = camera_folder

    def get_item(self):
        landmarks = []
        extrinsics = []
        intrinsics = []
        for frame in self.frames:
            landmarks_ = []
            extrinsics_ = []
            intrinsics_ = []
            camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, frame)))]
            for v in range(len(camera_ids)):
                if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v])):
                    landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
                    landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                    extrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['extrinsic']
                    intrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['intrinsic']
                else:
                    landmark = np.zeros([66, 3], dtype=np.float32)
                    extrinsic = np.ones([3, 4], dtype=np.float32)
                    intrinsic = np.ones([3, 3], dtype=np.float32)
                landmarks_.append(landmark)
                extrinsics_.append(extrinsic)
                intrinsics_.append(intrinsic)
            landmarks_ = np.stack(landmarks_)
            extrinsics_ = np.stack(extrinsics_)
            intrinsics_ = np.stack(intrinsics_)
            landmarks.append(landmarks_)
            extrinsics.append(extrinsics_)
            intrinsics.append(intrinsics_)
        landmarks = np.stack(landmarks)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return landmarks, extrinsics, intrinsics, self.frames
    
    def __len__(self):
        return len(self.frames)
    
    