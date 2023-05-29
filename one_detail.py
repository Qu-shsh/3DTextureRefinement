import torch.optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter

os.environ["PATH"] = os.environ["PATH"] + ":/opt/conda/bin/ninja"
from options.test_options import TestOptions
from models import create_model
from util.preprocess import align_img, estimate_norm
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d_faceverse

import cv2
from models.losses import perceptual_loss, photo_loss
from models.UNet import Unet, Unet_2, Unet_0
from models import networks

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
    save_path = "./checkpoints/FaceVerse_model_detail/continue/" + name.split(os.path.sep)[-1]
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(save_path))
        os.mkdir(os.path.join(save_path, "uv"))
        os.mkdir(os.path.join(save_path, "output"))
        os.mkdir(os.path.join(save_path, "mesh"))
        os.mkdir(os.path.join(save_path, "logs"))

    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)  # models.facerecon_model.FaceReconFVModel
    model.setup(opt)
    model.device = device

    model.net_recog = networks.define_net_recog(net_recog="r100",
                                          pretrained_path="./checkpoints/recog_model/ms1mv3_arcface_r100_fp16/backbone.pth")
    model.parallel_names += ['net_recog']

    model.parallelize()
    model.eval()

    im_path, lm_path, msk_path = get_data_path(name)

    lm3d_std = load_lm3d_faceverse(opt.faceverse_folder)

    UNet_final = Unet_0(6, 3).cuda()
    optimizer = torch.optim.Adam([{"params": UNet_final.parameters()}], lr=1e-3)

    # init_path = "./checkpoints/FaceVerse_model_detail/train/pth/epoch_2.pth"
    # state_dict = torch.load(init_path, map_location='cpu')
    # UNet_final.load_state_dict(state_dict["unet"])
    # optimizer.load_state_dict(state_dict["optim"])
    # print("loading init UNet and optimizer from %s" % (init_path))

    w_color = 1
    w_vertex = 0.8
    w_feat = 0.2

    # init tensorboard writer
    writer = SummaryWriter(os.path.join(save_path, "logs"))

    for epo in range(400):
        # for i in range(len(im_path)):
        for i in range(38,39):
            # print(i, im_path[i])

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

            model.get_uv()

            model.get_pos()
            uv_pos_tex = model.get_postex()

            # cv2.imwrite("./uvtest/uv_input.jpg", im[:, :, ::-1])
            # img = model.uv_shape[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            # cv2.imwrite("./uvtest/uv_shape.jpg", img[:, :, ::-1])
            # img = model.uv_texture[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            # cv2.imwrite("./uvtest/uv_tex.jpg", img[:,:,::-1])
            # img = model.uv_pos[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            # cv2.imwrite("./uvtest/uv_pos.jpg", img[:, :, ::-1])
            # img = uv_pos_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            # cv2.imwrite("./uvtest/uv_pos_tex.jpg", img[:,:,::-1])

            input_final = torch.cat((uv_pos_tex, model.uv_tex), dim=1)  # concat
            out_final = UNet_final(input_final)  # UNet输出
            out_final = out_final * model.facemodel.uv_mask
            pred_mask_detail, pred_face_detail = model.get_detail(model.uv_shape, out_final)

            optimizer.zero_grad()

            pred_feat = model.net_recog(model.pred_face_detail, model.trans_m)
            gt_feat = model.net_recog(model.input_img, model.trans_m)

            atten_mask = model.atten_mask
            zero = torch.zeros_like(atten_mask)
            one = torch.ones_like(atten_mask)
            atten_mask = torch.where(atten_mask >= opt.threshold, one, zero)

            loss1 = w_color * photo_loss(pred_face_detail, model.input_img, pred_mask_detail*atten_mask)
            loss2 = w_vertex * photo_loss(out_final, model.uv_tex, model.facemodel.uv_mask)
            loss3 = w_feat * perceptual_loss(pred_feat, gt_feat)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            if epo == 0 or (epo + 1) % 20 == 0:
                # 保存训练输出
                # tensorboard write loss
                writer.add_scalar("loss_color", loss1, (epo*100)+i+1)
                writer.add_scalar("loss_vertex", loss2, (epo*100)+i+1)
                writer.add_scalar("loss_feat", loss3, (epo*100)+i+1)
                writer.add_scalar("loss_all", loss, (epo*100)+i+1)
                # 打印loss
                print('This is img {i} in epoch {epo}, loss_all is {loss_all}, loss_color is {loss_color}, loss_vertex is {loss_vertex}, loss_feat is {loss_feat}'.format(i=i+1, epo=epo + 1, loss_color=loss1, loss_vertex=loss2, loss_feat=loss3, loss_all=loss))

            # 保存图像
            if epo==0 or (epo+1) % 100 == 0:
                img_input = im_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255
                uv_input = model.uv_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                pr_output = uv_pos_tex[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                final_output = out_final[0].permute(1, 2, 0).detach().cpu().numpy() * 255

                pred_mask_detail_base, pred_face_detail_base = model.get_detail(model.uv_shape, model.uv_tex)
                model.save_mesh(os.path.join(save_path, "mesh", img_name + "_base.obj"))
                img_output_base = pred_face_detail_base * pred_mask_detail_base + (1 - pred_mask_detail_base) * model.input_img
                img_output_base = img_output_base[0].permute(1, 2, 0).detach().cpu().numpy() * 255

                pred_mask_detail_uvpos, pred_face_detail_uvpos = model.get_detail(model.uv_shape, uv_pos_tex)
                img_output_uvpos = pred_face_detail_uvpos * pred_mask_detail_uvpos + (1 - pred_mask_detail_uvpos) * model.input_img
                img_output_uvpos = img_output_uvpos[0].permute(1, 2, 0).detach().cpu().numpy() * 255

                pred_mask_detail_final, pred_face_detail_final = model.get_detail(model.uv_shape, out_final)
                model.save_detail_mesh(os.path.join(save_path, "mesh", img_name + "_" + str(epo+1) + "_detail.obj"))
                img_output_final = pred_face_detail_final * pred_mask_detail_final + (1 - pred_mask_detail_final) * model.input_img
                img_output_final = img_output_final[0].permute(1, 2, 0).detach().cpu().numpy() * 255

                output_uv = np.concatenate((uv_input, pr_output, final_output), axis=-2)
                output_vis = np.concatenate((img_input, img_output_base, img_output_uvpos, img_output_final), axis=-2)
                cv2.imwrite(os.path.join(save_path, "uv", img_name + "_" + str(epo+1) + ".jpg"), output_uv[:, :, ::-1])
                cv2.imwrite(os.path.join(save_path, "output", img_name + "_" + str(epo+1) + ".jpg"), output_vis[:, :, ::-1])
    writer.close()

if __name__ == '__main__':
    opt = TestOptions().parse()
    main(0, opt, opt.img_folder)
