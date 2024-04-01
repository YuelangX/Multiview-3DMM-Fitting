import os
from yacs.config import CfgNode as CN
 


class config():

    def __init__(self):

        self.cfg = CN()
        self.cfg.image_folder = ''
        self.cfg.camera_folder = ''
        self.cfg.landmark_folder = ''
        self.cfg.param_folder = ''
        self.cfg.gpu_id = 0
        self.cfg.camera_ids = []
        self.cfg.image_size = 512
        self.cfg.face_model = 'BFM'
        self.cfg.reg_id_weight = 1e-6
        self.cfg.reg_exp_weight = 1e-6
        self.cfg.visualize = False
        self.cfg.save_vertices = False


    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self, config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()
