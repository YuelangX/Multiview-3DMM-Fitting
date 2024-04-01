import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from einops import rearrange
from pytorch3d.io import load_obj
from pytorch3d.transforms import so3_exponential_map
from smplx.lbs import batch_rigid_transform, vertices2joints, blend_shapes, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

from typing import NewType
Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def get_normal_coord_system(normals):
    """
    returns tensor of basis vectors of coordinate system that moves with normals:

    e_x = always points horizontally
    e_y = always pointing up
    e_z = normal vector

    returns tensor of shape N x 3 x 3

    :param normals: tensor of shape N x 3
    :return:
    """
    device = normals.device
    dtype = normals.dtype
    N = len(normals)

    assert len(normals.shape) == 2

    normals = normals.detach()
    e_y = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    e_x = torch.tensor([0., 0., 1.], device=device, dtype=dtype)

    basis = torch.zeros(len(normals), 3, 3, dtype=dtype, device=device)
    # e_z' = e_n
    basis[:, 2] = torch.nn.functional.normalize(normals, p=2, dim=-1)

    # e_x' = e_n x e_y except e_n || e_y then e_x' = e_x
    normal_parallel_ey_mask = ((basis[:, 2] * e_y[None]).sum(dim=-1).abs() == 1)
    basis[:, 0] = torch.cross(e_y.expand(N, 3), basis[:, 2], dim=-1)
    basis[normal_parallel_ey_mask][:, 0] = e_x[None]
    basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)
    basis[normal_parallel_ey_mask][:, 0] = e_x[None]
    basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)

    # e_y' = e_z' x e_x'
    basis[:, 1] = torch.cross(basis[:, 2], basis[:, 0], dim=-1)
    #basis[:, 1] = torch.nn.functional.normalize(basis[:, 1], p=2, dim=-1)

    assert torch.all(torch.norm(basis, dim=-1, p=2) > .99)

    return basis

def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    faces: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True
):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    
    normals = vertex_normals(v_posed, faces)
    B, V, _3 = normals.shape
    normal_coord_sys = get_normal_coord_system(normals.view(-1, 3)).view(B, V, 3, 3)


    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, batch_size):

        self.shape_dims = 100
        self.exp_dims = 50

        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open('assets/FLAME/generic_model.pkl', 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.NECK_IDX = 1
        self.batch_size = batch_size
        self.dtype = torch.float32
        
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project 
        default_shape = torch.zeros([self.batch_size, 300 - self.shape_dims],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                      requires_grad=False))

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project 
        default_exp = torch.zeros([self.batch_size, 100 - self.exp_dims],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                            requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open('assets/FLAME/flame_static_embedding.pkl', 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))


        conture_embeddings = np.load('assets/FLAME/flame_dynamic_embedding.npy',
            allow_pickle=True, encoding='latin1')
        conture_embeddings = conture_embeddings[()]
        dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
        dynamic_lmk_faces_idx = torch.tensor(
            dynamic_lmk_faces_idx,
            dtype=torch.long)
        self.register_buffer('dynamic_lmk_faces_idx',
                                dynamic_lmk_faces_idx)

        dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
        dynamic_lmk_bary_coords = torch.tensor(
            dynamic_lmk_bary_coords, dtype=self.dtype)
        self.register_buffer('dynamic_lmk_bary_coords',
                                dynamic_lmk_bary_coords)

        neck_kin_chain = []
        curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain',
                                torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx,
                                         dynamic_lmk_b_coords,
                                         neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=vertices.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params, expression_params, pose_params, neck_pose, eye_pose):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        betas = torch.cat([shape_params,self.shape_betas, expression_params, self.expression_betas], dim=1)
        full_pose = torch.cat([pose_params[:,:3], neck_pose, pose_params[:,3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        faces = self.faces_tensor.unsqueeze(0).repeat(shape_params.shape[0], 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               faces, self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents, self.lbs_weights, )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            vertices, full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)

        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat(
            [dyn_lmk_bary_coords, lmk_bary_coords], 1)
        
        lmk_faces_idx = torch.cat([lmk_faces_idx[:, 0:48], lmk_faces_idx[:, 49:54], lmk_faces_idx[:, 55:68]], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords[:, 0:48], lmk_bary_coords[:, 49:54], lmk_bary_coords[:, 55:68]], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        return vertices, landmarks


class FLAMEModule(nn.Module):
    def __init__(self, batch_size):
        super(FLAMEModule, self).__init__()

        self.shape_dims = 100
        self.exp_dims = 50
        self.batch_size = batch_size

        self.flame = FLAME(batch_size)

        self.id_coeff = nn.Parameter(torch.zeros(1, self.shape_dims, dtype=torch.float32))
        self.exp_coeff = nn.Parameter(torch.zeros(self.batch_size, self.exp_dims + 9, dtype=torch.float32)) # include expression_params, jaw_pose, eye_pose
        self.pose = nn.Parameter(torch.zeros(batch_size, 6, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(1, 1, dtype=torch.float32))

        self.register_buffer('neck_pose', torch.zeros(self.batch_size, 3, dtype=torch.float32)) # not optimized
        self.register_buffer('global_rotation', torch.zeros(self.batch_size, 3, dtype=torch.float32)) # not optimized
        self.register_buffer('faces', self.flame.faces_tensor)

    def forward(self):
        expression_params = self.exp_coeff[:, : self.exp_dims]
        jaw_rotation = self.exp_coeff[:, self.exp_dims: self.exp_dims + 3]
        neck_pose = self.neck_pose
        eye_pose = self.exp_coeff[:, self.exp_dims + 3: self.exp_dims + 9]

        pose_params = torch.cat([self.global_rotation, jaw_rotation], 1)
        shape_params = self.id_coeff.repeat(self.batch_size, 1)
        vertices, landmarks = self.flame(shape_params, 
                                         expression_params, 
                                         pose_params, 
                                         neck_pose, 
                                         eye_pose)
        R = so3_exponential_map(self.pose[:, :3])
        T = self.pose[:, 3:]

        vertices = torch.bmm(vertices * self.scale, R.permute(0,2,1)) + T[:, None, :]
        landmarks = torch.bmm(landmarks * self.scale, R.permute(0,2,1)) + T[:, None, :]
        return vertices, landmarks

    def reg_loss(self, id_weight, exp_weight):
        id_reg_loss = (self.id_coeff ** 2).sum()
        exp_reg_loss = (self.exp_coeff[:, : self.exp_dims] ** 2).sum(-1).mean()
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