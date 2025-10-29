from __future__ import print_function
import os
import ast
import cv2
import torch
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm

from util import input_param, init_rtvec_test
from module import ProST

device = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 10000
ITER_STEPS = 150
BATCH_SIZE = 1


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_selection(start, end):
    return np.random.choice(np.concatenate((np.arange(-end, -start+1), np.arange(start, end+1))))


def main(args):
    actual_image_path_list = sorted(glob(f'{args.CT_dir}/*.nii'))
    segmentation_path_list = sorted(glob(f'{args.CTSeg_dir}/*.nii'))

    # TODO 1: no rotation and translation
    for i in tqdm(range(len(actual_image_path_list))):
        cadaver_id = actual_image_path_list[i].split('/')[-1].split('.')[0]
        os.makedirs(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}', exist_ok=True)

        CT_PATH = actual_image_path_list[i]
        SEG_PATH = segmentation_path_list[i]

        # Calculate geometric parameters
        param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE)

        # Initialize projection model
        projmodel = ProST(param).to(device)

        ########## Hard Code test groundtruth and initialize poses ##########
        # [rx, ry, rz, tx, ty, tz]
        random_rotation = np.array([0.0, 0.0, 0.0])
        random_translation = np.array([0.0, 0.0, 0.0])

        manual_rtvec_gt = np.array([[0.0 + random_rotation[0], 0.0 + random_rotation[1], 0.0 + random_rotation[2], 0.0 + random_translation[0], 0.0 + random_translation[1], 0.0 + random_translation[2]]])
        manual_rtvec_smp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # Apply random rotation and translation to the groundtruth pose
        image_id = f'{cadaver_id}_{int(random_rotation[0])}_{int(random_rotation[1])}_{int(random_rotation[2])}_{int(random_translation[0])}_{int(random_translation[1])}_{int(random_translation[2])}.png'
        
        manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3]*PI/180
        manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:]/norm_factor
        
        transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(device, manual_test=True, manual_rtvec_gt=manual_rtvec_gt, manual_rtvec_smp=manual_rtvec_smp)
        with torch.no_grad():
            target = projmodel(_3D_vol, ray_proj_mov, rtvec_gt, corner_pt)
            min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
            target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
            cv2.imwrite(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}/{image_id}', target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size)*255)

    # TODO 2: baseline
    if args.data_type == 'baseline':
        for i in range(len(actual_image_path_list)):
            cadaver_id = actual_image_path_list[i].split('/')[-1].split('.')[0]
            print(f'Processing cadaver {cadaver_id}')
            for _ in tqdm(range(args.num_drr)):
                os.makedirs(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}', exist_ok=True)

                CT_PATH = actual_image_path_list[i]
                SEG_PATH = segmentation_path_list[i]

                # Calculate geometric parameters
                param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE)

                # Initialize projection model
                projmodel = ProST(param).to(device)

                ########## Hard Code test groundtruth and initialize poses ##########
                # [rx, ry, rz, tx, ty, tz]
                random_rotation = np.array([0.0, 0.0, 0.0])
                random_translation = np.array([0.0, 0.0, 0.0])

                random_rotation = [random_selection(5, 20) for _ in range(3)]
                random_translation = [random_selection(5, 30) for _ in range(3)]

                manual_rtvec_gt = np.array([[0.0 + random_rotation[0], 0.0 + random_rotation[1], 0.0 + random_rotation[2], 0.0 + random_translation[0], 0.0 + random_translation[1], 0.0 + random_translation[2]]])
                manual_rtvec_smp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                # Apply random rotation and translation to the groundtruth pose
                image_id = f'{cadaver_id}_{int(random_rotation[0])}_{int(random_rotation[1])}_{int(random_rotation[2])}_{int(random_translation[0])}_{int(random_translation[1])}_{int(random_translation[2])}.png'
                
                manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3]*PI/180
                manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:]/norm_factor
                
                transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(device, manual_test=True, manual_rtvec_gt=manual_rtvec_gt, manual_rtvec_smp=manual_rtvec_smp)
                with torch.no_grad():
                    target = projmodel(_3D_vol, ray_proj_mov, rtvec_gt, corner_pt)
                    min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                    target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
                    cv2.imwrite(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}/{image_id}', target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size)*255)

    if args.data_type == 'hard':
        # TODO 3: hard
        for i in range(len(actual_image_path_list)):
            cadaver_id = actual_image_path_list[i].split('/')[-1].split('.')[0]
            print(f'Processing cadaver {cadaver_id}')
            for _ in tqdm(range(args.num_drr)):
                os.makedirs(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}', exist_ok=True)

                CT_PATH = actual_image_path_list[i]
                SEG_PATH = segmentation_path_list[i]

                # Calculate geometric parameters
                param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE)

                # Initialize projection model
                projmodel = ProST(param).to(device)

                ########## Hard Code test groundtruth and initialize poses ##########
                # [rx, ry, rz, tx, ty, tz]
                random_rotation = np.array([0.0, 0.0, 0.0])
                random_translation = np.array([0.0, 0.0, 0.0])

                random_rotation = [random_selection(20, 45) for _ in range(3)]
                random_translation = [random_selection(30, 50) for _ in range(3)]

                manual_rtvec_gt = np.array([[0.0 + random_rotation[0], 0.0 + random_rotation[1], 0.0 + random_rotation[2], 0.0 + random_translation[0], 0.0 + random_translation[1], 0.0 + random_translation[2]]])
                manual_rtvec_smp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                # Apply random rotation and translation to the groundtruth pose
                image_id = f'{cadaver_id}_{int(random_rotation[0])}_{int(random_rotation[1])}_{int(random_rotation[2])}_{int(random_translation[0])}_{int(random_translation[1])}_{int(random_translation[2])}.png'
                
                manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3]*PI/180
                manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:]/norm_factor
                
                transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(device, manual_test=True, manual_rtvec_gt=manual_rtvec_gt, manual_rtvec_smp=manual_rtvec_smp)
                with torch.no_grad():
                    target = projmodel(_3D_vol, ray_proj_mov, rtvec_gt, corner_pt)
                    min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                    target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
                    cv2.imwrite(f'{args.DRR_dir}_{args.data_type}/{cadaver_id}/{image_id}', target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size)*255)

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Arguemtn \"%s\" is not a list" % (s))
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--CT_dir', type=str, default='../data/CT', help='Directory containing CT images')
    parser.add_argument('--CTSeg_dir', type=str, default='../data/CTSeg', help='Directory containing CT segmentation images')
    parser.add_argument('--data_type', type=str, default='baseline', help='Type of data to generate: baseline or hard')
    parser.add_argument('--num_drr', type=int, default=1000, help='Number of DRR images to generate per cadaver for baseline data')
    parser.add_argument('--DRR_dir', type=str, default='../data_projected', help='Directory to save the DRR images')

    args = parser.parse_args()

    fix_seed(42)

    main(args)
