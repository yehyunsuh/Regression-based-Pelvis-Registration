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
        ([-24, -1, 29], [41, 38, -17]),
        ([25, 21, -38], [40, -21, -49]),
        ([22, -26, -13], [-12, 45, -45]),
        ([14, -2, -5], [-40, -32, 35]),
        ([18, 29, -25], [39, -18, -24]),
        ([-24, -44, 39], [28, 22, 29]),
        ([10, -24, -36], [-8, 44, 46]),
        ([31, -43, 30], [8, 20, -13]),
        ([-8, 19, 3], [40, -2, 0]),
        ([-38, 0, -4], [14, 33, 6]),
        ([11, 17, 14], [-30, -35, -2]),
        ([-11, 15, -39], [34, 38, -30]),
        ([1, 14, 31], [-42, -31, -13]),
        ([-45, -43, 11], [43, 7, -28]),
        ([25, 43, -7], [20, 44, -26]),
        ([-40, -27, 11], [18, 27, -40]),
        ([42, -4, 6], [15, -12, 5]),
        ([18, 17, 23], [-16, 12, -7]),
        ([-28, 19, -42], [-44, 42, -50]),
        ([-6, 5, -6], [-50, 38, 23]),
        ([20, 14, 20], [-50, -17, -37]),
        ([-7, -14, -34], [3, -6, 17]),
        ([-29, 8, -2], [34, 2, 43]),
        ([-12, -30, -30], [28, -32, -41]),
        ([-37, -21, 44], [-25, 44, 9]),
        ([31, -26, -26], [-39, 8, -47]),
        ([-4, -16, -19], [13, 33, -49]),
        ([-24, -29, 10], [0, 22, 2]),
        ([30, -32, 42], [27, 41, 33]),
        ([0, -1, -36], [-41, -17, 6]),
        ([16, -33, -41], [7, -38, 16]),
        ([25, -43, 28], [6, -41, 30]),
        ([-41, 36, 9], [-18, 37, 35]),
        ([38, 31, 5], [1, 6, 15]),
        ([-18, 23, -35], [12, 44, -13]),
        ([-7, 21, 3], [40, -37, -30]),
        ([27, 36, -25], [15, 29, -34]),
        ([-37, -21, 21], [6, -2, -25]),
        ([13, 2, 35], [50, 40, 10]),
        ([-29, -3, -10], [-8, 14, 29]),
        ([43, -1, -45], [30, 44, -11]),
        ([-13, 35, 30], [-16, -30, -33]),
        ([-29, -23, -42], [2, -14, 35]),
        ([37, -7, 21], [2, -33, 38]),
        ([-18, -32, -5], [-42, -23, 0]),
        ([36, 32, 27], [-37, 49, -25]),
        ([-45, 16, -7], [-32, 32, 46]),
        ([10, -30, -15], [-37, -11, 27]),
        ([25, -3, -20], [-29, 33, 48]),
        ([-16, 29, -15], [-33, -28, -9]),
        ([44, -28, 28], [-26, 15, 50]),
        ([26, 14, 3], [-27, -3, -47]),
        ([-17, -19, 41], [7, -11, -9]),
        ([-33, 17, 24], [-16, -19, -24]),
        ([-6, 17, -20], [-36, -45, 0]),
        ([-27, -28, 4], [-48, -39, -13]),
        ([-39, 26, 29], [-21, -3, 24]),
        ([19, -11, -14], [-11, 21, 37]),
        ([-34, -20, -12], [29, -26, -50]),
        ([-38, -21, 9], [-30, -38, 49]),
        ([18, -38, 21], [-14, -11, -19]),
        ([-36, -31, 20], [-23, -18, -36]),
        ([-36, -38, -43], [40, -9, -41]),
        ([-35, 9, 43], [-8, 29, 10]),
        ([19, -27, 22], [31, -49, 4]),
        ([-8, 26, -39], [34, 47, -22]),
        ([40, -19, -10], [-33, 18, 29]),
        ([-8, -7, 3], [44, 31, -49]),
        ([10, -3, 23], [-47, -41, -48]),
        ([42, 22, -21], [0, -14, 33]),
        ([-26, -22, -6], [47, 12, -4]),
        ([18, 41, 25], [-47, 47, -31]),
        ([-23, 29, -16], [1, -11, 28]),
        ([-11, 11, 26], [42, -38, 39]),
        ([14, 6, -19], [-20, -43, -36]),
        ([-22, 2, -32], [-2, -43, -11]),
        ([-21, -13, -27], [-24, -9, -8]),
        ([29, -25, -40], [-8, 9, 3]),
        ([-19, 16, -6], [-48, 6, -6]),
        ([-7, -6, 15], [50, -34, 39]),
        ([27, 45, 1], [31, -35, -47]),
        ([-16, -7, -38], [30, 48, -7]),
        ([37, -20, 21], [-11, -41, 28]),
        ([16, 39, -1], [27, -43, 7]),
        ([-34, 0, -17], [46, 5, 43]),
        ([-39, 28, -11], [17, 36, 19]),
        ([-28, 38, -23], [30, -3, 29]),
        ([15, -43, -22], [46, -3, 39]),
        ([39, 27, 6], [-29, 30, 5]),
        ([22, -34, 40], [40, 37, 38]),
        ([-3, 36, 34], [-7, 7, 12]),
        ([6, 17, -17], [39, -15, -49]),
        ([9, -14, -25], [-16, -29, -26]),
        ([-6, -33, -15], [-19, 38, 30]),
        ([-2, 33, 24], [38, 17, -3]),
        ([-35, 15, 40], [-47, 17, -19]),
        ([-40, 3, 14], [14, 1, 33]),
        ([4, 35, 29], [43, -5, -7]),
        ([12, -6, 40], [-17, 20, -2]),
        ([-30, -19, 16], [28, -24, -48]),
    ]
    
    train(args, rotation_translation_pairs, 'hard')
