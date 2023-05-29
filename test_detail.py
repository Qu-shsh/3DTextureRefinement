import torch.optim
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from torch.utils.tensorboard import SummaryWriter

os.environ["PATH"] = os.environ["PATH"] + ":/opt/conda/bin/ninja"
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img, estimate_norm
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d_faceverse

import cv2
from models.UNet import Unet
import time

def get_data_path(root):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    msk_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'mask', i.split(os.path.sep)[-1]) for i in im_path]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in
               lm_path]

    return im_path, lm_path, msk_path

def read_data(im_path, lm_path, msk_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size

    msk = Image.open(msk_path).convert('RGB')

    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    _, im_crop, lm, mask = align_img(im, lm, lm3d_std, msk)  # 裁剪、对齐
    # im_crop.save("test_input.jpg")
    # mask.save("./uvtest/mask.jpg")
    M = estimate_norm(lm, im_crop.size[1])

    if to_tensor:
        im_ten = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        im_crop_ten = torch.tensor(np.array(im_crop) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        msk_ten = torch.tensor(np.array(mask)/255., dtype=torch.float32).permute(2, 0, 1)[:1, ...].unsqueeze(0)
        lm_ten = torch.tensor(lm).unsqueeze(0)
        M_ten = torch.tensor(np.array(M).astype(np.float32)).unsqueeze(0)
    return im_ten, im_crop_ten, msk_ten, lm_ten, M_ten

def main(rank, opt, name):

    save_path = "./checkpoints/FaceVerse_model_detail/test/" + name.split(os.path.sep)[-1]
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(save_path))
        os.mkdir(os.path.join(save_path, "uv"))
        os.mkdir(os.path.join(save_path, "output"))
        os.mkdir(os.path.join(save_path, "mesh"))

    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)  # models.facerecon_model.FaceReconFVModel
    model.setup(opt)
    model.device = device

    model.parallelize()
    model.eval()

    # total_num = sum(p.numel() for p in model.net_recon.parameters())

    im_path, lm_path, msk_path = get_data_path(name)

    lm3d_std = load_lm3d_faceverse(opt.faceverse_folder)

    UNet_final = Unet(6, 3).cuda()
    init_path = "./checkpoints/FaceVerse_model_detail/train/pth/epoch_20.pth"
    state_dict = torch.load(init_path, map_location='cpu')
    UNet_final.load_state_dict(state_dict["unet"])
    print("loading init UNet from %s" % (init_path))

    # total_num = sum(p.numel() for p in UNet_final.parameters())

    for i in range(len(im_path)):
    # for i in range(5,6):
        start = time.time()

        img_name = im_path[i].split(os.path.sep)[-1].replace('.png', '').replace('.jpg', '')

        if not os.path.isfile(lm_path[i]):
            continue

        imraw_tensor, im_tensor, msk_tensor, lm_tensor, M_tensor = read_data(im_path[i], lm_path[i], msk_path[i], lm3d_std)
        data = {
            'test_vid': im_tensor,
            'msks': msk_tensor,
            'lms': lm_tensor,
            'M': M_tensor,
            'raw': imraw_tensor
        }

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        mid = time.time()
        print('img %d , path %s , 1 Time Taken: %.3f ms.' % (i, im_path[i], (mid - start) * 1000))

        model.get_uv()

        model.get_pos()
        uv_pos_tex = model.get_postex()

        # cv2.imwrite("./uvtest/uv_input.jpg", im[:, :, ::-1])
        # img = model.uv_shape[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite("./uvtest/uv_shape.jpg", img[:, :, ::-1])
        # img = model.uv_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite("./uvtest/uv_tex.jpg", img[:,:,::-1])
        # img = model.uv_pos[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite("./uvtest/uv_pos.jpg", img[:,:,::-1])
        # img = uv_pos_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite("./uvtest/uv_pos_tex.jpg", img[:,:,::-1])

        input_final = torch.cat((uv_pos_tex, model.uv_tex), dim=1)  # concat
        out_final = UNet_final(input_final)  # UNet输出
        out_final = out_final * model.facemodel.uv_mask

        end = time.time()
        print('img %d , path %s , 2 Time Taken: %.3f ms.' % (i, im_path[i], (end - mid)*1000))

        # 保存图像
        img_input = im_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255
        uv_input = model.uv_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        pr_output = uv_pos_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        final_output = out_final[0].permute(1, 2, 0).detach().cpu().numpy() * 255

        pred_mask_detail_base, pred_face_detail_base = model.get_detail(model.uv_shape, model.uv_tex)
        # model.save_mesh(os.path.join(save_path, "mesh", img_name + "_base.obj"))
        img_output_base = pred_face_detail_base * pred_mask_detail_base + (1 - pred_mask_detail_base) * model.input_img
        img_output_base = img_output_base[0].permute(1, 2, 0).detach().cpu().numpy() * 255

        pred_mask_detail_uvpos, pred_face_detail_uvpos = model.get_detail(model.uv_shape, uv_pos_tex)
        img_output_uvpos = pred_face_detail_uvpos * pred_mask_detail_uvpos + (1 - pred_mask_detail_uvpos) * model.input_img
        img_output_uvpos = img_output_uvpos[0].permute(1, 2, 0).detach().cpu().numpy() * 255

        pred_mask_detail_final, pred_face_detail_final = model.get_detail(model.uv_shape, out_final)
        # model.save_detail_mesh(os.path.join(save_path, "mesh", img_name + "_detail.obj"))
        img_output_final = pred_face_detail_final * pred_mask_detail_final + (1 - pred_mask_detail_final) * model.input_img
        img_output_final = img_output_final[0].permute(1, 2, 0).detach().cpu().numpy() * 255

        output_uv = np.concatenate((uv_input, pr_output, final_output), axis=-2)
        output_vis = np.concatenate((img_input, img_output_base, img_output_uvpos, img_output_final), axis=-2)
        cv2.imwrite(os.path.join(save_path, "uv", img_name + ".jpg"), output_uv[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, "output", img_name + ".jpg"), output_vis[:, :, ::-1])

if __name__ == '__main__':
    opt = TestOptions().parse()
    main(0, opt, opt.img_folder)
