import torch
from torch import nn
import numpy as np


class FaceVerseModel(nn.Module):
    def __init__(self, keypoints, model_dict, front_face_buf, keypoints_detail, detail_dict, batch_size=1, focal=1015,
                 img_size=224,
                 use_simplification=False, device='cuda:0'):
        super(FaceVerseModel, self).__init__()

        self.focal = focal
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device

        self.p_mat = self._get_p_mat(device)
        self.reverse_z = self._get_reverse_z(device)
        self.camera_pos = self._get_camera_pose(device)
        self.camera_distance = 10.0
        self.rotXYZ = torch.eye(3).view(1, 3, 3).repeat(3, 1, 1).view(3, 1, 3, 3).to(self.device)

        if use_simplification:
            self.select_id = model_dict['select_id']
            self.select_id_tris = np.vstack(
                (self.select_id * 3, self.select_id * 3 + 1, self.select_id * 3 + 2)).transpose().flatten()
            self.skinmask = torch.tensor(model_dict['skinmask_select'], requires_grad=False, device=self.device)

            self.kp_inds = torch.tensor(model_dict['keypoints_select'].reshape(-1, 1),
                                        requires_grad=False).squeeze().long().to(self.device)

            self.meanshape = torch.tensor(model_dict['meanshape'].reshape(1, -1)[:, self.select_id_tris],
                                          dtype=torch.float32, requires_grad=False, device=self.device)
            self.meantex = torch.tensor(model_dict['meantex'].reshape(1, -1)[:, self.select_id_tris],
                                        dtype=torch.float32, requires_grad=False, device=self.device)

            self.idBase = torch.tensor(model_dict['idBase'][self.select_id_tris], dtype=torch.float32,
                                       requires_grad=False, device=self.device)
            self.expBase = torch.tensor(model_dict['exBase'][self.select_id_tris], dtype=torch.float32,
                                        requires_grad=False, device=self.device)
            self.texBase = torch.tensor(model_dict['texBase'][self.select_id_tris], dtype=torch.float32,
                                        requires_grad=False, device=self.device)

            self.tri = torch.tensor(model_dict['tri_select'], dtype=torch.int64, requires_grad=False,
                                    device=self.device)
            self.point_buf = torch.tensor(model_dict['point_buf_select'], dtype=torch.int64, requires_grad=False,
                                          device=self.device)

        else:
            # Base
            self.skin_mask = torch.tensor(model_dict['skinmask'][0], requires_grad=False, device=self.device)

            # self.kp_inds = torch.tensor(model_dict['keypoints'].reshape(-1, 1), requires_grad=False).squeeze().long().to(self.device)
            self.kp_inds = torch.tensor(keypoints.reshape([-1, 1]), requires_grad=False).squeeze().long().to(
                self.device)
            # self.kp_inds = torch.tensor(keypoints_detail.reshape([-1, 1]), requires_grad=False).squeeze().long().to(self.device)

            self.meanshape = torch.tensor(model_dict['meanshape'].reshape(1, -1) * 1.5, dtype=torch.float32,
                                          requires_grad=False, device=self.device)
            self.meantex = torch.tensor(model_dict['meantex'].reshape(1, -1), dtype=torch.float32, requires_grad=False,
                                        device=self.device)

            self.idBase = torch.tensor(model_dict['idBase'], dtype=torch.float32, requires_grad=False,
                                       device=self.device)
            self.expBase = torch.tensor(model_dict['exBase'], dtype=torch.float32, requires_grad=False,
                                        device=self.device)
            self.texBase = torch.tensor(model_dict['texBase'], dtype=torch.float32, requires_grad=False,
                                        device=self.device)
            self.uv_base = torch.tensor(np.hstack((model_dict['uv'], np.zeros((model_dict['uv'].shape[0], 1)))),
                                        dtype=torch.long, requires_grad=False, device=self.device)

            self.tri = torch.tensor(model_dict['tri'], dtype=torch.int64, requires_grad=False, device=self.device)
            self.point_buf = torch.tensor(model_dict['point_buf'], dtype=torch.int64, requires_grad=False,
                                          device=self.device)
            self.uv_mask = torch.tensor(model_dict['uvmask'], dtype=torch.int64, requires_grad=False,
                                               device=self.device)
            self.front_face_buf = torch.tensor(front_face_buf, dtype=torch.int64, requires_grad=False,
                                               device=self.device)

            # Detail
            self.uv_detail = torch.tensor(detail_dict['uv'], dtype=torch.long, requires_grad=False,
                                          device=self.device)
            self.tri_detail = torch.tensor(detail_dict['tri'], dtype=torch.int64, requires_grad=False,
                                           device=self.device)
            self.point_buf_detail = torch.tensor(detail_dict['point_buf'], dtype=torch.int64, requires_grad=False,
                                                 device=self.device)
            self.uv_mask_detail = torch.tensor(detail_dict['uvmask'], dtype=torch.int64, requires_grad=False,
                                        device=self.device)

        self.num_vertex = self.meanshape.shape[1] // 3
        self.id_dims = self.idBase.shape[1]
        self.tex_dims = self.texBase.shape[1]
        self.exp_dims = self.expBase.shape[1]
        self.all_dims = self.id_dims + self.tex_dims + self.exp_dims

        # for tracking by landmarks
        # self.kp_inds_view = torch.cat(
        #     [self.kp_inds[:, None] * 3, self.kp_inds[:, None] * 3 + 1, self.kp_inds[:, None] * 3 + 2], dim=1).flatten()
        # self.idBase_view = self.idBase[self.kp_inds_view, :].detach().clone()
        # self.expBase_view = self.expBase[self.kp_inds_view, :].detach().clone()
        # self.meanshape_view = self.meanshape[:, self.kp_inds_view].detach().clone()

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff
        exp_coeff = coeffs[:, self.id_dims:self.id_dims + self.exp_dims]  # expression coeff
        tex_coeff = coeffs[:, self.id_dims + self.exp_dims:self.all_dims]  # texture(albedo) coeff
        angles = coeffs[:, self.all_dims:self.all_dims + 3]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeffs[:, self.all_dims + 3:self.all_dims + 30]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeffs[:, self.all_dims + 30:]  # translation coeff of dim 3

        return {
            'id': id_coeff,
            'exp': exp_coeff,
            'tex': tex_coeff,
            'angle': angles,
            'gamma': gamma,
            'trans': translation
        }

    def to_camera(self, face_shape):
        face_shape[..., 1] = -face_shape[..., 1]
        face_shape[..., 2] = -face_shape[..., 2]
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def get_vs(self, id_coeff, exp_coeff):
        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape
        face_shape = face_shape.view(self.batch_size, -1, 3)
        return face_shape

    # def get_vs_lms(self, id_coeff, exp_coeff):
    #     face_shape = torch.einsum('ij,aj->ai', self.idBase_view, id_coeff) + \
    #                  torch.einsum('ij,aj->ai', self.expBase_view, torch.abs(exp_coeff)) + self.meanshape_view
    #     face_shape = face_shape.view(self.batch_size, -1, 3)
    #     return face_shape

    def get_color(self, tex_coeff):
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex
        face_texture = face_texture / 255.
        face_texture = face_texture.view(self.batch_size, -1, 3)
        return face_texture

    def get_skinmask(self):
        return self.skinmask

    def _get_camera_pose(self, device):
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=device).reshape(1, 1, 3)
        return camera_pos

    def _get_p_mat(self, device):
        half_image_width = self.img_size // 2
        p_matrix = np.array([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return torch.tensor(p_matrix, device=device)

    def _get_reverse_z(self, device):
        reverse_z = np.reshape(np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3])
        return torch.tensor(reverse_z, device=device)

    def compute_norm(self, vs, tri, point_buf):
        face_id = tri
        point_id = point_buf
        v1 = vs[:, face_id[:, 0], :]
        v2 = vs[:, face_id[:, 1], :]
        v3 = vs[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)

        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-9)

        return v_norm

    def project_vs(self, vs):
        # vs = torch.matmul(vs, self.reverse_z.repeat((self.batch_size, 1, 1)))+self.camera_pos
        aug_projection = torch.matmul(vs, self.p_mat.repeat((self.batch_size, 1, 1)).permute((0, 2, 1)))
        face_projection = aug_projection[:, :, :2] / torch.reshape(aug_projection[:, :, 2], [self.batch_size, -1, 1])
        return face_projection

    def compute_rotation_matrix(self, angles):
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        if self.batch_size != 1:
            rotXYZ = self.rotXYZ.repeat(1, self.batch_size, 1, 1)
        else:
            rotXYZ = self.rotXYZ.detach().clone()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def add_illumination(self, face_texture, norm, gamma):
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8
        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(a0 * c0 * (nx * 0 + 1))
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(self.batch_size, face_texture.shape[1], 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """

        # Base
        coef_dict = self.split_coeffs(coeffs)
        face_shape = self.get_vs(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation_matrix(coef_dict['angle'])

        face_vertex = self.rigid_transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_vertex)
        landmark = self.get_lms(face_vertex)
        landmark = self.project_vs(landmark)
        # landmark = torch.stack([landmark[:, :, 0], self.img_size - landmark[:, :, 1]], dim=2)

        face_texture = self.get_color(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape, self.tri, self.point_buf)
        face_norm_roted = face_norm.bmm(rotation)
        face_color = self.add_illumination(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark

    def touv(self, coef_dict):
        shape = self.get_vs(coef_dict['id'], coef_dict['exp'])
        color = self.get_color(coef_dict['tex'])

        uv_geo = torch.tensor(np.zeros((self.batch_size, 256, 256, 3), np.float32), device=self.device)
        uv_tex = torch.tensor(np.zeros((self.batch_size, 256, 256, 3), np.float32), device=self.device)

        uv_geo[:, self.uv_base[:, 1] + 28, self.uv_base[:, 0] + 28] = shape
        uv_tex[:, self.uv_base[:, 1] + 28, self.uv_base[:, 0] + 28] = color

        # uv detail
        # uv_geo = nn.functional.interpolate(uv_geo.permute(0, 3, 1, 2), scale_factor=4, mode='bilinear', align_corners=False)
        # uv_tex = nn.functional.interpolate(uv_tex.permute(0, 3, 1, 2), scale_factor=4, mode='bilinear', align_corners=False)

        # uv base
        uv_geo = uv_geo.permute(0, 3, 1, 2)
        uv_tex = uv_tex.permute(0, 3, 1, 2)

        return uv_geo, uv_tex

    def topos(self, uv_shape, coef_dict):
        # uv detail
        # shape = uv_shape[:, :, self.uv_detail[:, 1], self.uv_detail[:, 0]].permute(0, 2, 1)

        # uv base
        shape = uv_shape[:, :, self.uv_base[:, 1]+28, self.uv_base[:, 0]+28].permute(0, 2, 1)

        rotation = self.compute_rotation_matrix(coef_dict['angle'])
        face_vertex = self.rigid_transform(shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_vertex)

        K = self.perspective_projection(1315, 112)
        transformed_vertices = face_vertex @ K
        transformed_vertices = transformed_vertices / transformed_vertices[..., 2:]

        # uv detail
        # uv_position_map = torch.tensor(np.zeros((self.batch_size, 3, 1024, 1024), np.float32), device=self.device)
        # uv_position_map[:, :, self.uv_detail[:, 1], self.uv_detail[:, 0]] = transformed_vertices.permute(0, 2, 1)
        # uv_position_map = uv_position_map/255

        # uv base
        uv_position_map = torch.tensor(np.zeros((self.batch_size, 3, 256, 256), np.float32), device=self.device)
        uv_position_map[:, :, self.uv_base[:, 1]+28, self.uv_base[:, 0]+28] = transformed_vertices.permute(0, 2, 1)
        uv_position_map = uv_position_map/255

        return uv_position_map

    def todetail(self, uv_geo, uv_tex, coef_dict):
        # uv detail
        # face_shape = uv_geo[:, :, self.uv_detail[:, 1], self.uv_detail[:, 0]].permute(0, 2, 1)
        # face_texture = uv_tex[:, :, self.uv_detail[:, 1], self.uv_detail[:, 0]].permute(0, 2, 1)

        # uv base
        face_shape = uv_geo[:, :, self.uv_base[:, 1]+28, self.uv_base[:, 0]+28].permute(0, 2, 1)
        face_texture = uv_tex[:, :, self.uv_base[:, 1]+28, self.uv_base[:, 0]+28].permute(0, 2, 1)

        rotation = self.compute_rotation_matrix(coef_dict['angle'])
        face_vertex = self.rigid_transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_vertex)

        landmark = self.get_lms(face_vertex)
        landmark = self.project_vs(landmark)

        # uv detail
        # face_norm = self.compute_norm(face_shape, self.tri_detail, self.point_buf_detail)
        # uv base
        face_norm = self.compute_norm(face_shape, self.tri, self.point_buf)

        face_norm_roted = face_norm.bmm(rotation)
        face_color = self.add_illumination(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark

    def perspective_projection(self, focal, center):
        return torch.tensor(np.array([
            focal, 0, center,
            0, focal, center,
            0, 0, 1
        ]).reshape([3, 3]).astype(np.float32).transpose(), device=self.device)
