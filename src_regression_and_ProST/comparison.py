from __future__ import print_function
import cv2 
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import ast
import imageio.v2 as imageio
import os
import shutil
import torch.nn.functional as F
import time

from tqdm import tqdm
from io import BytesIO
from glob import glob

from util import ncc, input_param, init_rtvec_test, gradncc
from util_plot import plot_example_regi
from module import ProST

from classification_input1 import test_model as test_model_input1
from classification_input1_PE import test_model as test_model_input1_PE
from classification_input1_PE_AC import test_model as test_model_input1_PE_AC
from classification_input2 import test_model as test_model_input2
from classification_input2_PE import test_model as test_model_input2_PE
from classification_input2_PE_AC import test_model as test_model_input2_PE_AC

DEVICE = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 10000
ITER_STEPS = 300
BATCH_SIZE = 1

CT_PATH = sorted(glob('../NMDID_test/CT/*.nii.gz'))[24]
SEG_PATH = sorted(glob('../NMDID_test/CT_seg/*.nii.gz'))[24]

case_id = CT_PATH.split('/')[-1].split('.')[0]
print('\n===============================================')
print("Current Case ID: ", case_id)

# Intensity - Medium: R_-5_-10_14_T_12_-20_-1
# Intensity - Difficult: R_14_-2_-5_T_-40_-32_35
# MICCAI - Medium: R_-11_-13_20_T_15_26_5
# MICCAI - Difficult:  R_11_17_14_T_-30_-35_-2
# TMI - Medium:  R_1_7_11_T_7_-14_-30
# TMI - Difficult:  R_-37_-21_44_T_-25_44_9
# Intensity
intensity_rotation_translation_pairs = [
    # [[-5, -10, 14], [12, -20, -1]],
    [[-11, -13, 20], [15, 26, 5]],
    [[1, 7, 11], [7, -14, -30]],
    # R_16_16_19_T_-4_8_-27
    [[16, 16, 19], [-4, 8, -27]],
]


initializer = False
os.makedirs('comparison', exist_ok=True)
for i in range(len(intensity_rotation_translation_pairs)):
    image_name = f'R_{int(intensity_rotation_translation_pairs[i][0][0])}_{int(intensity_rotation_translation_pairs[i][0][1])}_{int(intensity_rotation_translation_pairs[i][0][2])}_T_{int(intensity_rotation_translation_pairs[i][1][0])}_{int(intensity_rotation_translation_pairs[i][1][1])}_{int(intensity_rotation_translation_pairs[i][1][2])}'
    criterion_gradncc = gradncc
    param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE)
    projmodel = ProST(param).to(DEVICE)

    print()
    print(f'Experiment {i}: Rotation: {intensity_rotation_translation_pairs[i][0]}, Translation: {intensity_rotation_translation_pairs[i][1]}')

    manual_rotation = np.array(intensity_rotation_translation_pairs[i][0])
    manual_translation = np.array(intensity_rotation_translation_pairs[i][1])

    manual_rtvec_gt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # [rx, ry, rz, tx, ty, tz]
    print(f'Ground: {manual_rtvec_gt[0]}')
    manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3] * PI / 180
    manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:] / norm_factor

    manual_rtvec_smp_mov = np.array([[0.0 + manual_rotation[0], 0.0 + manual_rotation[1], 0.0 + manual_rotation[2], 0.0 + manual_translation[0], 0.0 + manual_translation[1], 0.0 + manual_translation[2]]])
    print(f'Sample: {manual_rtvec_smp_mov[0]}')
    manual_rtvec_smp_mov[:, :3] = manual_rtvec_smp_mov[:, :3] * PI / 180
    manual_rtvec_smp_mov[:, 3:] = manual_rtvec_smp_mov[:, 3:] / norm_factor

    _, rtvec_mov, rtvec_gt = init_rtvec_test(
        DEVICE, 
        manual_test=True,
        manual_rtvec_gt=manual_rtvec_gt,
        manual_rtvec_smp=manual_rtvec_smp_mov
    )

    with torch.no_grad():
        gt = projmodel(_3D_vol, ray_proj_mov, rtvec_gt, corner_pt)
        min_gt, _ = torch.min(gt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_gt, _ = torch.max(gt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        gt = (gt.reshape(BATCH_SIZE, -1) - min_gt) / (max_gt - min_gt)
        gt = gt.reshape(BATCH_SIZE, 1, det_size, det_size)
        # cv2.imwrite(f'comparison/Intensity_{image_name}_gt.png', gt[0,0,:,:].detach().cpu().numpy() * 255)

        target = projmodel(_3D_vol, ray_proj_mov, rtvec_mov, corner_pt)  # rtvecgt -> rtvec_mov
        min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
        target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
        cv2.imwrite(f'comparison/Intensity_{image_name}_target.png', target[0,0,:,:].detach().cpu().numpy() * 255)


    if initializer:
        prediction = test_model_input1(None, target, DEVICE, image_name, 'Intensity')
    else:
        prediction = np.array([[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]])

    if len(prediction[0]) == 6:
        manual_rtvec_smp = np.array([[0.0+prediction[0,0], 0.0+prediction[0,1], 0.0+prediction[0,2], 0.0+prediction[0,3], 0.0+prediction[0,4], 0.0+prediction[0,5]]])
    else:
        manual_rtvec_smp = np.array([[0.0, 0.0, 0.0, 0.0 + prediction[0,0], 0.0 + prediction[0,1], 0.0 + prediction[0,2]]])
    print(f'Predic: {manual_rtvec_smp[0]}')
    manual_rtvec_smp[:, :3] = manual_rtvec_smp[:, :3]*PI/180
    manual_rtvec_smp[:, 3:] = manual_rtvec_smp[:, 3:]/norm_factor
    
    _, rtvec, _ = init_rtvec_test(
        DEVICE, 
        manual_test=True,
        manual_rtvec_gt=manual_rtvec_gt,
        manual_rtvec_smp=manual_rtvec_smp
    )

    with torch.no_grad():
        pred = projmodel(_3D_vol, ray_proj_mov, rtvec, corner_pt)
        min_pred, _ = torch.min(pred.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_pred, _ = torch.max(pred.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        pred = (pred.reshape(BATCH_SIZE, -1) - min_pred) / (max_pred - min_pred)
        pred = pred.reshape(BATCH_SIZE, 1, det_size, det_size)
        if initializer:
            cv2.imwrite(f'comparison/Intensity_{image_name}_pred_init.png', pred[0,0,:,:].detach().cpu().numpy() * 255)
        else:
            cv2.imwrite(f'comparison/Intensity_{image_name}_pred.png', pred[0,0,:,:].detach().cpu().numpy() * 255)

    # Use Pytorch SGD optimizer
    optimizer_gradncc = optim.SGD([rtvec], lr=0.002, momentum=0.9)
    gradncc_sim_list = []
    rtvec_diff_list = []

    stop = False
    convergence = False
    start_time, end_time, conv_iter = time.time(), None, None
    count = 0
    # for iter in tqdm(range(ITER_STEPS)):
    for iter in range(ITER_STEPS):
        if iter > 10:
            stop = np.std(gradncc_sim_list[-10:]) < 1e-4

        rtvec_diff = rtvec.detach().cpu()[0,:]-rtvec_mov.detach().cpu()[0,:]
        rtvec_diff_list.append(rtvec_diff.detach().cpu().numpy())
        proj_mov = projmodel(_3D_vol, target, rtvec, corner_pt)

        optimizer_gradncc.zero_grad()

        gradncc_loss = criterion_gradncc(proj_mov, target)

        if gradncc_loss.item() >= 1:
            count += 1

        gradncc_sim_list.append(gradncc_loss.item())
        gradncc_loss.backward()
        optimizer_gradncc.step()

        if stop:
            convergence = True
            end_time = time.time()
            conv_iter = iter
            break

        if gradncc_loss.item() >= 1:
            count += 1

        if count > 10:
            break

        print(f'Iter: {iter}, GradNCC Similarity: {gradncc_loss.item()}')
    
    angle_x = rtvec[0][0].detach().cpu().numpy() * 180 / PI
    angle_y = rtvec[0][1].detach().cpu().numpy() * 180 / PI
    angle_z = rtvec[0][2].detach().cpu().numpy() * 180 / PI
    trans_x = rtvec[0][3].detach().cpu().numpy() * norm_factor
    trans_y = rtvec[0][4].detach().cpu().numpy() * norm_factor
    trans_z = rtvec[0][5].detach().cpu().numpy() * norm_factor

    print(f'Angle: {angle_x}, {angle_y}, {angle_z}')
    print(f'Trans: {trans_x}, {trans_y}, {trans_z}')

    manual_rtvec_smp_mov = np.array([[angle_x, angle_y, angle_z, trans_x, trans_y, trans_z]])
    print(f'Optim: {manual_rtvec_smp_mov[0]}')
    manual_rtvec_smp_mov[:, :3] = manual_rtvec_smp_mov[:, :3] * PI / 180
    manual_rtvec_smp_mov[:, 3:] = manual_rtvec_smp_mov[:, 3:] / norm_factor

    _, rtvec_opt, rtvec_gt = init_rtvec_test(
        DEVICE, 
        manual_test=True,
        manual_rtvec_gt=manual_rtvec_gt,
        manual_rtvec_smp=manual_rtvec_smp_mov
    )

    with torch.no_grad():
        opt = projmodel(_3D_vol, ray_proj_mov, rtvec_opt, corner_pt)
        min_opt, _ = torch.min(opt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_opt, _ = torch.max(opt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        opt = (opt.reshape(BATCH_SIZE, -1) - min_opt) / (max_opt - min_opt)
        opt = opt.reshape(BATCH_SIZE, 1, det_size, det_size)
        if initializer:
            cv2.imwrite(f'comparison/Intensity_{image_name}_opt_init.png', opt[0,0,:,:].detach().cpu().numpy() * 255)
        else:
            cv2.imwrite(f'comparison/Intensity_{image_name}_opt.png', opt[0,0,:,:].detach().cpu().numpy() * 255)
