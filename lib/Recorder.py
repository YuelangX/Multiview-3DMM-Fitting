import torch
import numpy as np
import os
import cv2
import trimesh
import pyrender


class Recorder():
    def __init__(self, save_folder, camera, visualize=False, save_vertices=False):

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.camera = camera

        self.visualize = visualize
        self.save_vertices = save_vertices

    def log(self, log_data):
        frames = log_data['frames']
        face_model = log_data['face_model']
        intrinsics = log_data['intrinsics']
        extrinsics = log_data['extrinsics']
     
        with torch.no_grad():
            vertices, landmarks = log_data['face_model']()
        
        for n, frame in enumerate(frames):
            os.makedirs(os.path.join(self.save_folder, frame), exist_ok=True)
            face_model.save('%s/params.npz' % (os.path.join(self.save_folder, frame)), batch_id=n)
            np.save('%s/lmk_3d.npy' % (os.path.join(self.save_folder, frame)), landmarks[n].cpu().numpy())
            if self.save_vertices:
                np.save('%s/vertices.npy' % (os.path.join(self.save_folder, frame)), vertices[n].cpu().numpy())

            if self.visualize:
                for v in range(intrinsics.shape[1]):
                    faces = log_data['face_model'].faces.cpu().numpy()
                    mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
                    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

                    self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    render_image = self.camera.render(mesh)

                    cv2.imwrite('%s/vis_%d.jpg' % (os.path.join(self.save_folder, frame), v), render_image[:,:,::-1])
                