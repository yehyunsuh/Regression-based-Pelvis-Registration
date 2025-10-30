from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from glob import glob
import cv2
import os
import argparse
import time
import torch.nn.functional as F

import numpy as np
from torch.optim.lr_scheduler import StepLR, CyclicLR

from module import ProST_init, Pelvis_Dataset
from module_vit import RegiNet_CrossViTv2_SW
from util import gradncc, init_rtvec_test, input_param
from util_plot import plot_test_iter_comb

from geomstats.geometry.special_euclidean import SpecialEuclidean

from classification_input1 import test_model as test_model_input1
from classification_input1_PE import test_model as test_model_input1_PE
from classification_input1_PE_AC import test_model as test_model_input1_PE_AC
from classification_input2 import test_model as test_model_input2
from classification_input2_PE import test_model as test_model_input2_PE
from classification_input2_PE_AC import test_model as test_model_input2_PE_AC


device = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 10000
BATCH_SIZE = 1
ITER_STEPS = 300

MANUAL_TEST = True

# SE3_GROUP = SpecialEuclidean(n=3)
SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
# METRIC = SE3_GROUP.left_canonical_metric
METRIC = SE3_GROUP.metric

VOX_SPAC = 2.33203125

lr_net = 0.001
lr_gradncc = 0.001
switch_trd = 0.003
stop_trd = 1e-4
zFlip = False


def prerequisite(args, rotation, translation):
    args.wandb_name = f'R_{int(rotation[0])}_{int(rotation[1])}_{int(rotation[2])}_T_{int(translation[0])}_{int(translation[1])}_{int(translation[2])}'
    if 'Baseline' in args.wandb_project:
        args.vis_dir = f'visualization/ProST/'
    elif 'Reg' in args.wandb_project:
        args.vis_dir = f'visualization/Reg/'
    else:
        args.vis_dir = f'visualization/Meta/'

    if 'Input1' in args.model_type:
        if 'PE' in args.model_type and 'AC' not in args.model_type:
            args.vis_dir += 'classification_ver1_PE'
        elif 'PE' in args.model_type and 'AC' in args.model_type:
            args.vis_dir += 'classification_ver1_PE_AC'
        else:
            args.vis_dir += 'classification_ver1'
    elif 'Input2' in args.model_type:
        if 'PE' in args.model_type and 'AC' not in args.model_type:
            args.vis_dir += 'classification_ver2_PE'
        elif 'PE' in args.model_type and 'AC' in args.model_type:
            args.vis_dir += 'classification_ver2_PE_AC'
        else:
            args.vis_dir += 'classification_ver2'
    else:
        args.vis_dir += 'baseline'

    args.vis_dir += f'/{args.wandb_name}'
    # os.makedirs('iterfigs', exist_ok=True)

    return args


def train(args, rotation_translation_pairs, experiment_type):
    SAVE_PATH = '../data/save_model'
    RESUME_MODEL = SAVE_PATH+f'/{args.model_registration}.pt'

    CT_PATH_list = sorted(glob('../NMDID/CT/*.nii.gz'))[20:25]
    SEG_PATH_list = sorted(glob('../NMDID/CTseg/*.nii.gz'))[20:25]

    for CT_PATH, SEG_PATH in zip(CT_PATH_list, SEG_PATH_list):
        case_id = CT_PATH.split('/')[-1].split('.')[0]
        print('\n===============================================')
        print("Current Case ID: ", case_id)

        for i in range(len(rotation_translation_pairs)):
            args = prerequisite(args, rotation_translation_pairs[i][0], rotation_translation_pairs[i][1])

            criterion_mse = nn.MSELoss()
            criterion_gradncc = gradncc
            param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE, VOX_SPAC, zFlip)

            initmodel = ProST_init(param).to(device)
            model = RegiNet_CrossViTv2_SW().to(device)

            checkpoint = torch.load(RESUME_MODEL, map_location=device)
            model.load_state_dict(checkpoint['model-state-dict'])

            model.eval()
            model.require_grad = False
            
            print()
            print(f'Experiment {i}: Rotation: {rotation_translation_pairs[i][0]}, Translation: {rotation_translation_pairs[i][1]}')

            manual_rotation = np.array(rotation_translation_pairs[i][0])
            manual_translation = np.array(rotation_translation_pairs[i][1])

            # Get target  projection
            if MANUAL_TEST:
                manual_rtvec_gt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # [rx, ry, rz, tx, ty, tz]
                print(f'Ground: {manual_rtvec_gt[0]}')
                manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3] * PI / 180
                manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:] / norm_factor

                manual_rtvec_smp_mov = np.array([[0.000001 + manual_rotation[0], 0.000001 + manual_rotation[1], 0.000001 + manual_rotation[2], 0.000001 + manual_translation[0], 0.000001 + manual_translation[1], 0.000001 + manual_translation[2]]])
                print(f'Sample: {manual_rtvec_smp_mov[0][0]:.2f}, {manual_rtvec_smp_mov[0][1]:.2f}, {manual_rtvec_smp_mov[0][2]:.2f}, {manual_rtvec_smp_mov[0][3]:.2f}, {manual_rtvec_smp_mov[0][4]:.2f}, {manual_rtvec_smp_mov[0][5]:.2f}')
                manual_rtvec_smp_mov[:, :3] = manual_rtvec_smp_mov[:, :3] * PI / 180
                manual_rtvec_smp_mov[:, 3:] = manual_rtvec_smp_mov[:, 3:] / norm_factor

            else:
                manual_rtvec_gt = None
                manual_rtvec_smp = None

            transform_mat3x4_gt, rtvec_mov, rtvec_gt = init_rtvec_test(device, manual_test=True, manual_rtvec_gt=manual_rtvec_gt, manual_rtvec_smp=manual_rtvec_smp_mov)
            transform_mat3x4_mov, rtvec_mov, rtvec_gt = init_rtvec_test(device, manual_test=True, manual_rtvec_gt=manual_rtvec_smp_mov, manual_rtvec_smp=manual_rtvec_smp_mov)

            with torch.no_grad():
                gt = initmodel(CT_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt)
                min_tar, _ = torch.min(gt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(gt.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                gt = (gt.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                gt = gt.reshape(BATCH_SIZE, 1, det_size, det_size)

                gt_numpy = (gt[0,0,:,:].detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f'tmp/{args.wandb_project}_gt_{experiment_type}.png', gt_numpy)

                target = initmodel(CT_vol, ray_proj_mov, transform_mat3x4_mov, corner_pt)
                min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

                target_numpy = (target[0,0,:,:].detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f'tmp/{args.wandb_project}_target_{experiment_type}.png', target_numpy)
            
            start_time = time.time()
            if 'Input1' in args.model_type:
                if 'PE' in args.model_type and 'AC' not in args.model_type:
                    prediction = test_model_input1_PE(args, target, device, experiment_type)
                elif 'PE' in args.model_type and 'AC' in args.model_type:
                    prediction = test_model_input1_PE_AC(args, target, device, experiment_type)
                else:
                    prediction = test_model_input1(args, target, device, experiment_type)
            elif 'Input2' in args.model_type:
                if 'PE' in args.model_type and 'AC' not in args.model_type:
                    prediction = test_model_input2_PE(args, target, gt, device, experiment_type)
                elif 'PE' in args.model_type and 'AC' in args.model_type:
                    prediction = test_model_input2_PE_AC(args, target, gt, device, experiment_type)
                else:
                    prediction = test_model_input2(args, target, gt, device, experiment_type)
            else:
                prediction = np.array([[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]])

            if len(prediction[0]) == 6:
                manual_rtvec_gt = np.array([[0.0+prediction[0,0], 0.0+prediction[0,1], 0.0+prediction[0,2], 0.0+prediction[0,3], 0.0+prediction[0,4], 0.0+prediction[0,5]]])
            else:
                manual_rtvec_gt = np.array([[0.0, 0.0, 0.0, 0.0 + prediction[0,0], 0.0 + prediction[0,1], 0.0 + prediction[0,2]]])
                
            print(f'Predic: {manual_rtvec_gt[0][0]:.2f}, {manual_rtvec_gt[0][1]:.2f}, {manual_rtvec_gt[0][2]:.2f}, {manual_rtvec_gt[0][3]:.2f}, {manual_rtvec_gt[0][4]:.2f}, {manual_rtvec_gt[0][5]:.2f}')
            manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3]*PI/180
            manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:]/norm_factor

            transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(
                device, 
                manual_test=True,
                manual_rtvec_gt=manual_rtvec_gt,
                manual_rtvec_smp=manual_rtvec_smp_mov
            )

            with torch.no_grad():
                target = initmodel(CT_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt)
                min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

                target_numpy = (target[0,0,:,:].detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f'tmp/{args.wandb_project}_target_init.png', target_numpy)

            optimizer_net = optim.SGD([rtvec], lr=lr_net, momentum=0.9)
            optimizer_gradncc = optim.SGD([rtvec], lr=lr_gradncc, momentum=0.9)
            #scheduler_net = CyclicLR(optimizer_gradncc, base_lr=0.005, max_lr=0.02,step_size_up=20)
            scheduler_gradncc = CyclicLR(optimizer_gradncc, base_lr=0.001, max_lr=0.003,step_size_up=20)

            network_sim_list = []
            gradncc_sim_list = []
            rtvec_diff_list = []

            stop = False
            switch = False
            convergence = False
            end_time, conv_iter = time.time(), None
            count = 0
            for iter in range(ITER_STEPS):
                if not switch:
                    if iter > 10:
                        network_sim_list_np = np.array(network_sim_list)
                        switch = np.std(network_sim_list_np[-10:]) < switch_trd
                    else:
                        switch = False

                if switch and iter>10:
                    stop = np.std(gradncc_sim_list[-10:]) < stop_trd

                vals, proj_mov = model(_3D_vol, target, rtvec, corner_pt, param)

                encoder_share_weights = True
                if type(vals) == bool and (not vals):# and (not vals[0])
                    continue
                elif len(vals) == 3 and vals[-1] and (not encoder_share_weights):
                    encode_mov = vals[0]
                    encode_tar = vals[1]
                elif len(vals) == 2 and vals[-1] and encoder_share_weights:
                    encode_out = vals[0]
                else:
                    print('Invalid Model Return!!!!!')
                    continue

                optimizer_net.zero_grad()
                optimizer_gradncc.zero_grad()

                # Network Similarity:
                l2_loss = torch.mean(encode_out) if encoder_share_weights else criterion_mse(encode_mov, encode_tar) #RegiNet loss
                # gradncc Similarity:
                gradncc_loss = criterion_gradncc(proj_mov, target)

                network_sim_list.append(l2_loss.item())
                gradncc_sim_list.append(gradncc_loss.item())

                rtvec.retain_grad()

                if switch:
                    gradncc_loss.backward()
                    optimizer_gradncc.step()
                    scheduler_gradncc.step()
                else:
                    l2_loss.backward()
                    optimizer_net.step()

                rtvec_diff = rtvec.detach().cpu()[0,:]-rtvec_gt.detach().cpu()[0,:]
                rtvec_diff_list.append(rtvec_diff.detach().cpu().numpy())

                # if iter == 0:
                #     proj_init_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())

                # if iter % 5 == 0:
                #     fig = plt.figure(figsize=(15, 9))
                #     plot_test_iter_comb(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                #         network_sim_list, gradncc_sim_list, rtvec_diff_list, switch)
            
                # print('Iter: %d, Network Sim: %.4f, GradNCC Sim: %.4f' % (iter, l2_loss.item(), gradncc_loss.item()))

                if stop:
                    convergence = True
                    end_time = time.time()
                    conv_iter = iter
                    break

                if gradncc_loss.item() >= 1 or l2_loss.item() < 0:
                    count += 1

                if count > 10:
                    break

            if not convergence:
                end_time = time.time()
                conv_iter = ITER_STEPS

            # Save the best results in csv file
            file_path = f'csv/{args.wandb_project}_{experiment_type}.csv'

            # The column name should follow what I save in wandb
            column_names = [
                'Case ID', 'Experiment Name', 'Iteration', 'Time',

                'Initial GradNCC Similarity',
                'Initial Rotation X Diff', 'Initial Rotation Y Diff', 'Initial Rotation Z Diff', 
                'Initial Translation X Diff', 'Initial Translation Y Diff', 'Initial Translation Z Diff', 

                'GradNCC Similarity', 
                'Final Rotation X', 'Final Rotation Y', 'Final Rotation Z', 
                'Final Translation X', 'Final Translation Y', 'Final Translation Z', 

                'MAE Angle Difference', 'MAE Translation Difference', 
                # 'MSE Difference'
            ]

            # print(f'Angle Diff: {rtvec_diff_list[-1][0] * 180 / PI}, {rtvec_diff_list[-1][1] * 180 / PI}, {rtvec_diff_list[-1][2] * 180 / PI}')
            # print(f'Translation Diff: {rtvec_diff_list[-1][3] * norm_factor}, {rtvec_diff_list[-1][4] * norm_factor}, {rtvec_diff_list[-1][5] * norm_factor}')

            mae_angle_diff = np.mean(abs(rtvec_diff_list[-1][0] * 180 / PI) + abs(rtvec_diff_list[-1][1] * 180 / PI) + abs(rtvec_diff_list[-1][2] * 180 / PI))
            mae_translation_diff = np.mean(abs(rtvec_diff_list[-1][3] * norm_factor) + abs(rtvec_diff_list[-1][4] * norm_factor) + abs(rtvec_diff_list[-1][5] * norm_factor))

            if os.path.exists(file_path):
                with open(file_path, 'a') as f:
                    f.write(
                        f'{case_id},'
                        f'{args.wandb_name},'
                        f'{conv_iter},'
                        f'{end_time - start_time},'

                        f'{gradncc_sim_list[0]},'
                        f'{abs(rtvec_diff_list[0][0] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][1] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][2] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][3] * norm_factor)},'
                        f'{abs(rtvec_diff_list[0][4] * norm_factor)},'
                        f'{abs(rtvec_diff_list[0][5] * norm_factor)},'

                        f'{gradncc_sim_list[-1]},'
                        f'{abs(rtvec_diff_list[-1][0] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][1] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][2] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][3] * norm_factor)},'
                        f'{abs(rtvec_diff_list[-1][4] * norm_factor)},'
                        f'{abs(rtvec_diff_list[-1][5] * norm_factor)},'
                        
                        f'{mae_angle_diff},'
                        f'{mae_translation_diff}\n'
                        # f'{mse_diff.item()}\n'
                    )

            else:
                with open(file_path, 'a') as f:
                    f.write(','.join(column_names) + '\n')
                    f.write(
                        f'{case_id},'
                        f'{args.wandb_name},'
                        f'{conv_iter},'
                        f'{end_time - start_time},'

                        f'{gradncc_sim_list[0]},'
                        f'{abs(rtvec_diff_list[0][0] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][1] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][2] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[0][3] * norm_factor)},'
                        f'{abs(rtvec_diff_list[0][4] * norm_factor)},'
                        f'{abs(rtvec_diff_list[0][5] * norm_factor)},'

                        f'{gradncc_loss.item()},'
                        f'{abs(rtvec_diff_list[-1][0] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][1] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][2] * 180 / PI)},'
                        f'{abs(rtvec_diff_list[-1][3] * norm_factor)},'
                        f'{abs(rtvec_diff_list[-1][4] * norm_factor)},'
                        f'{abs(rtvec_diff_list[-1][5] * norm_factor)},'
                        
                        f'{mae_angle_diff},'
                        f'{mae_translation_diff}\n'
                        # f'{mse_diff.item()}\n'
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb_name', type=str, default='ProST_MICCAI')
    parser.add_argument('--wandb_project', type=str, default='ProST_MICCAI')
    
    parser.add_argument('--model_type', type=str, default='', help='Model type')
    parser.add_argument('--weight_path', type=str, default='../model_weight/GuessNet_Regression_easy.pth', help='Path to the classification model weights')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the classification model')

    parser.add_argument('--model_registration', type=str, default='pretrain', help='Model for registration')

    args = parser.parse_args()

    os.makedirs('csv', exist_ok=True)

    rotation_translation_pairs = [
        ([-20, -14, -16], [-21, -17, 7]),
        ([-15, -11, 18], [-18, -18, 28]),
        ([-11, -19, -10], [26, -15, 18]),
        ([-10, 18, -11], [-26, 1, 25]),
        ([-17, 14, -8], [-9, 28, 0]),
        ([8, -3, -15], [5, 22, 19]),
        ([-9, 16, -14], [-1, -22, 5]),
        ([-8, -10, 6], [27, -10, -10]),
        ([3, 19, 1], [22, -17, 22]),
        ([10, -3, -2], [-21, 25, 23]),
        ([15, 18, 8], [6, 29, -10]),
        ([13, -10, 19], [15, -19, -6]),
        ([20, -13, 2], [6, -18, -13]),
        ([-15, -3, 9], [-12, -17, -10]),
        ([10, -10, 17], [4, 0, 23]),
        ([-16, -20, -11], [-29, 17, 29]),
        ([16, 16, 19], [-4, 8, -27]),
        ([11, 9, 7], [7, 14, -28]),
        ([-7, -4, 20], [-4, 7, 5]),
        ([-3, -11, -17], [24, 14, -16]),
        ([-10, -19, 13], [-29, -4, -25]),
        ([9, 18, -16], [-26, -27, -27]),
        ([19, -3, 5], [0, 22, -12]),
        ([-9, 7, 10], [-30, -11, 14]),
        ([-17, -16, -7], [16, 11, 22]),
        ([6, 11, -9], [7, -23, -21]),
        ([17, -13, -6], [25, 28, 10]),
        ([-19, -6, -7], [-27, -5, -6]),
        ([-4, 18, 3], [3, -10, -24]),
        ([-15, 9, -7], [-21, -11, 29]),
        ([1, 5, -2], [-18, 1, 10]),
        ([-19, -7, -4], [-4, -28, 30]),
        ([-1, -16, -9], [-6, -10, 25]),
        ([20, -11, -7], [-7, -28, -26]),
        ([6, -12, -5], [0, -4, 19]),
        ([6, 19, -1], [13, 20, 16]),
        ([4, 14, 16], [-2, -18, -3]),
        ([-19, 17, -2], [16, 24, -19]),
        ([11, -18, -7], [22, 17, 3]),
        ([17, -15, -14], [12, -1, -4]),
        ([1, -5, -8], [21, 25, -25]),
        ([1, -16, 17], [0, -29, -22]),
        ([7, 2, -5], [-26, -20, 11]),
        ([9, -1, 3], [20, -12, -2]),
        ([1, -13, 0], [24, 19, -14]),
        ([-12, -4, 11], [28, -22, 2]),
        ([-4, -12, 19], [-12, -28, 15]),
        ([2, 2, -19], [12, 25, 6]),
        ([-14, 3, -18], [-16, 17, 11]),
        ([13, -10, -8], [30, 23, 3]),
        ([18, 4, 20], [30, -23, 12]),
        ([-5, 15, -1], [13, 7, -18]),
        ([3, -17, -20], [-15, 10, -22]),
        ([7, 12, 13], [4, -30, -28]),
        ([0, 10, 3], [-26, 7, 0]),
        ([-11, -13, 20], [15, 26, 5]),
        ([20, 20, -2], [-19, -30, -9]),
        ([-19, 15, -16], [-18, -29, -3]),
        ([7, 12, 19], [24, 27, -15]),
        ([2, 19, -4], [8, -18, 12]),
        ([-16, -4, -10], [5, 10, 28]),
        ([7, -1, -20], [-17, 26, 12]),
        ([-2, 0, -1], [-6, 11, -14]),
        ([1, 4, -7], [26, 25, -8]),
        ([4, 10, -14], [20, 28, 14]),
        ([-14, -19, 18], [-18, -19, 10]),
        ([-19, -19, 0], [5, 14, 10]),
        ([-8, 18, 14], [-16, 14, -17]),
        ([-19, -7, -5], [-29, -3, -19]),
        ([19, -3, 5], [16, -21, 23]),
        ([-14, -1, 20], [6, 12, 28]),
        ([0, 19, -20], [-4, -25, -21]),
        ([12, -16, 15], [1, -15, 21]),
        ([-2, -15, 17], [-17, -7, 20]),
        ([3, -7, -20], [10, -30, 23]),
        ([-8, 17, 2], [1, -18, 13]),
        ([1, 7, 11], [7, -14, -30]),
        ([2, 14, 1], [-25, -3, -8]),
        ([11, -4, -11], [11, 9, -6]),
        ([-13, -20, 20], [5, 4, 22]),
        ([1, -9, 16], [24, 23, 12]),
        ([16, 4, -8], [-22, -7, -19]),
        ([-2, 4, 11], [5, -3, 10]),
        ([-18, -19, 5], [9, -11, 21]),
        ([18, 7, -15], [18, 8, -29]),
        ([20, -18, -12], [10, -21, -21]),
        ([1, 9, 16], [26, 1, 12]),
        ([-17, 14, -20], [-14, -13, -11]),
        ([5, -3, 13], [13, 1, -12]),
        ([-12, 1, -6], [-20, 22, 8]),
        ([-5, -10, 14], [12, -20, -1]),
        ([15, -19, -10], [-27, 16, -20]),
        ([-4, 16, 3], [-9, 0, -11]),
        ([-18, -6, -10], [25, -8, -12]),
        ([20, 4, 17], [16, -14, 12]),
        ([5, -16, -5], [-7, -7, 18]),
        ([12, 14, 7], [23, -11, -6]),
        ([10, 5, -11], [-30, -30, 1]),
        ([1, -20, -20], [1, 3, 28]),
        ([17, -20, 5], [-11, 16, 23]),
    ]
    
    train(args, rotation_translation_pairs, 'base')
