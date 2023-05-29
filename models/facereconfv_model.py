import numpy as np
import os
import torch
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
from .FaceVerseModel import FaceVerseModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from util import util
from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch
from util.detect_lm68 import draw_landmarks

import trimesh
from scipy.io import savemat
import cv2

class FaceReconFVModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False,help='zero initialize the last fc')
        parser.add_argument('--faceverse_folder', type=str, default='FaceVerse')
        parser.add_argument('--faceverse_model', type=str, default='faceverse_base_v1.npy', help='faceverse model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1315.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5)
        parser.add_argument('--z_far', type=float, default=15)

        # mask threshold
        parser.add_argument('--threshold', type=float, default=0.7)

        if is_train:
            # training parameters
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50', 'r100'],help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str,default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False,help='use crop mask for photo loss')
            parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False,help='use predefined M for predicted face')

            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')

        opt, _ = parser.parse_known_args()
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)

        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path, model=opt.model
        )

        # 加载模型参数
        faceverse_dict = np.load(os.path.join(opt.faceverse_folder,opt.faceverse_model), allow_pickle=True).item()
        keypoints = np.loadtxt(os.path.join(opt.faceverse_folder,"Dlib68_index.txt")).astype(np.float32)
        front_face_buf = np.loadtxt(os.path.join(opt.faceverse_folder, "front_face_tri.txt")).astype(np.float32)
        faceverse_detail_dict = np.load(os.path.join(opt.faceverse_folder,opt.faceverse_model.replace("base","detail")), allow_pickle=True).item()
        keypoints_detail = np.loadtxt(os.path.join(opt.faceverse_folder, "Dlib68_index_detail.txt")).astype(np.float32)
        # 构建FaceVerse模型
        if self.isTrain:
            self.facemodel = FaceVerseModel(keypoints, faceverse_dict, front_face_buf, keypoints_detail, faceverse_detail_dict, batch_size=opt.batch_size, focal=opt.focal)
        else:
            self.facemodel = FaceVerseModel(keypoints, faceverse_dict, front_face_buf, keypoints_detail, faceverse_detail_dict, focal=opt.focal)

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        if self.isTrain:
            self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
            )
            # loss func name: (compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss
            self.comupte_color_loss = photo_loss
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = reflectance_loss

            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['test_vid'].to(self.device)
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device) if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None
        self.raw = input['raw'].to(self.device) if 'raw' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(self.pred_vertex, self.facemodel.tri, feat=self.pred_color)

        # import cv2
        # # transform = transforms.RandomVerticalFlip(p=1)
        # # self.pred_face = transform(self.pred_face)
        # img = self.pred_face[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # img_new = img
        # cv2.imwrite("output.jpg", img_new)
        # lms=self.pred_lm[0].detach().cpu().numpy().astype(np.int64)
        # for index, l in enumerate(lms):
        #     pt_pos = (l[0], img.shape[0]-1-l[1])
        #     cv2.circle(img_new, pt_pos, 1, (0, 225, 0), 2)
        #     # font = cv2.FONT_HERSHEY_SIMPLEX
        #     # cv2.putText(img_new, str(index + 1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("output_lm.jpg", img_new)
        # # self.pred_mask = transform(self.pred_mask)
        # # img = self.pred_mask[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # # img_new = img[:, :, ::-1]
        # # cv2.imwrite("output_mask.jpg", img_new)

        self.pred_coeffs_dict = self.facemodel.split_coeffs(output_coeff)

    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask

        # import cv2
        # output_vis = self.pred_face * face_mask
        # img = output_vis[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite("facemask.jpg", img)

        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)

            # output_vis = self.pred_face * face_mask
            # img = output_vis[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            # cv2.imwrite("facemask_use_crop.jpg", img)

        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask)

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)
        # self.loss_reflc = 0

        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma + self.loss_lm + self.loss_reflc

    def optimize_parameters(self, isTrain=True):
        self.forward()
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()
            self.loss_all.backward()
            self.optimizer.step()

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')

                output_vis_numpy = np.concatenate((input_img_numpy,
                                                   output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy,
                                                   output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                output_vis_numpy / 255., dtype=torch.float32
            ).permute(0, 3, 1, 2).to(self.device)

    def save_mesh(self, name):
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.tri.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                               vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_detail_mesh(self, name):
        recon_shape = self.pred_vertex_detail  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color_detail
        recon_color = recon_color.detach().cpu().numpy()[0]

        # uv detail
        # tri = self.facemodel.tri_detail.cpu().numpy()
        # uv base
        tri = self.facemodel.tri.cpu().numpy()

        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                               vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self, name):

        pred_coeffs = {key: self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:, :, 0], self.input_img.shape[2] - 1 - pred_lm[:, :, 1]],
                           axis=2)  # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name, pred_coeffs)

    def get_uv(self):
        self.uv_shape, self.uv_tex = self.facemodel.touv(self.pred_coeffs_dict)

    def get_pos(self):
        self.uv_pos = self.facemodel.topos(self.uv_shape, self.pred_coeffs_dict)

    def get_postex(self):
        im = self.input_img[0]
        B, h, w = 1, self.uv_pos.shape[2], self.uv_pos.shape[3]
        transform = transforms.RandomVerticalFlip(p=1)
        im = transform(im).unsqueeze(0)

        mask = self.atten_mask[0]
        # cv2.imwrite("./uvtest/mask.jpg", (mask.permute(1,2,0).detach().cpu().numpy())*255)
        mask =  transform(mask).unsqueeze(0)

        '''
        im ---> 输入图像
                tensor类型
                shape=[1,3,224,244]
                值的范围[0,1]
                
        uv_pos ---> uv position map
                    tensor类型
                    shape=[1,3,1024,1024]/[1,3,256,256]
                    值的范围[0,1]
        '''
        im=im.squeeze(0).permute(1,2,0).detach().cpu().numpy() # im.shape=[224,224,3], 值的范围是[0,1]
        uv_pos = self.uv_pos.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255 # uv_pos.shape=[1024,1024,3]/[256,256,3], 值的范围是[0,255]
        uv_pos_tex = cv2.remap(im, uv_pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        # cv2.imwrite("./uvtest/uv_pos_tex.jpg", uv_pos_tex[:, :, ::-1]*255)
        self.uv_pos_tex = torch.tensor(uv_pos_tex , dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
        uv_pos_color = self.uv_pos_tex[:,:,self.facemodel.uv_base[:, 1] + 28, self.facemodel.uv_base[:, 0] + 28].permute(0, 2, 1)

        #  去掉光照
        face_shape = self.facemodel.get_vs(self.pred_coeffs_dict['id'], self.pred_coeffs_dict['exp'])
        face_norm = self.facemodel.compute_norm(face_shape, self.facemodel.tri, self.facemodel.point_buf)
        rotation = self.facemodel.compute_rotation_matrix(self.pred_coeffs_dict['angle'])
        face_norm_roted = face_norm.bmm(rotation)
        uv_pos_color_delgamma = self.del_illumination(uv_pos_color, face_norm_roted, self.pred_coeffs_dict['gamma'])
        self.uv_pos_tex[:, :, self.facemodel.uv_base[:, 1] + 28, self.facemodel.uv_base[:, 0] + 28] = uv_pos_color_delgamma.permute(0, 2, 1)
        self.uv_pos_tex = torch.clamp(self.uv_pos_tex, 0, 1)
        uv_pos_tex = self.uv_pos_tex.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite("./uvtest/uv_pos_tex_delgamma.jpg", uv_pos_tex[:, :, ::-1]*255)
        uv_pos_tex *= 255

        # mask remap
        mask = mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        uv_pos_mask = cv2.remap(mask, uv_pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        # cv2.imwrite("./uvtest/uv_pos_mask.jpg", uv_pos_mask * 255)

        uv_pos_mask_3 = np.zeros((256,256,3)).astype(np.uint8)
        uv_pos_mask_3[uv_pos_mask >= self.opt.threshold] = 255
        uv_pos_mask_3[uv_pos_mask < self.opt.threshold] = 0
        # cv2.imwrite("./uvtest/uv_pos_mask_new.jpg", uv_pos_mask_3)
        if (np.sum(uv_pos_mask_3>0)) < 100:
            uv_pos_mask_3 = (uv_pos_mask*255).astype(np.uint8)
            # cv2.imwrite("./uvtest/uv_pos_mask_new.jpg", uv_pos_mask_3)


        # uv_tex
        uv_texture = self.uv_tex.squeeze(0).permute(1,2,0).detach().cpu().numpy() # shape=[256,256,3], 值的范围是[0,1]
        uv_texture=np.clip(uv_texture,0,1)
        # cv2.imwrite("./uvtest/uv_tex.jpg", uv_texture[:,:,::-1]*255)
        uv_texture *= 255

        new_uv_pos_tex = cv2.seamlessClone(uv_pos_tex.astype(np.uint8), uv_texture.astype(np.uint8), uv_pos_mask_3, (128,128), cv2.MIXED_CLONE)
        # cv2.imwrite("./uvtest/uv_pos_tex_new.jpg", new_uv_pos_tex[:,:,::-1])

        new_uv_pos_tex =torch.tensor(new_uv_pos_tex/255, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)

        return new_uv_pos_tex

    def del_illumination(self, face_color, norm, gamma):
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
        Y = H.view(self.facemodel.batch_size, face_color.shape[1], 9)
        lighting = Y.bmm(gamma)

        face_texture = face_color / lighting

        return face_texture

    def get_detail(self,uv_shape,uv_tex):
        self.pred_vertex_detail, self.pred_tex_detail, self.pred_color_detail, self.pred_lm_detail = self.facemodel.todetail(uv_shape,uv_tex,self.pred_coeffs_dict)

        # uv detail
        # self.pred_mask_detail, _, self.pred_face_detail = self.renderer(self.pred_vertex_detail, self.facemodel.tri_detail, feat=self.pred_color_detail)
        # uv base
        self.pred_mask_detail, _, self.pred_face_detail = self.renderer(self.pred_vertex_detail, self.facemodel.tri, feat=self.pred_color_detail)

        return self.pred_mask_detail, self.pred_face_detail