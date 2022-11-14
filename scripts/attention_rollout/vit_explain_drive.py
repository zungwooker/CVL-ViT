# Updated 2022-10-26 14:10
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import glob

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
import vit_explain as vex

def minmax_ego(PATH_sample_data_list, drive_name):
    voxel_feature_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    for PATH_data in PATH_sample_data_list:
        with open(PATH_data, 'rb') as f:
            data = pickle.load(f)

        fm = data['feature_map']
        for x in range(13, 15):
            for y in range(13, 15):
                for key in voxel_feature_dict.keys():
                    voxel_feature_dict[key].append(fm[key][x][y])

    min_ego = {key:min(voxel_feature_dict[key]) for key in voxel_feature_dict.keys()}
    max_ego = {key:max(voxel_feature_dict[key]) for key in voxel_feature_dict.keys()}

    f = open(drive_name + '_ego.txt', 'w')
    f.write("MIN\n")
    for key in voxel_feature_dict.keys():
        f.write(key+': '+str(min_ego[key]))
        f.write('\n')
    f.write('\n')
    f.write("MAX\n")
    for key in voxel_feature_dict.keys():
        f.write(key+': '+str(max_ego[key]))
        f.write('\n')
    f.close()


def minmax_target_sample1(PATH_sample1_data_list, drive_name):
    # WARNING: HARD CODE
    file_number = [("00299", "00309")]
    minmax_file_number = file_number[0]

    voxel_feature_min_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    voxel_feature_max_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    f = open(drive_name + '_target.txt', 'w')

    for PATH_data in PATH_sample1_data_list:

        # MIN ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[0]:
            f.write("** MIN Attention: " + minmax_file_number[0]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(8, 10):
                for y in range(14, 16):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])

            min_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            min_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(min_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(min_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')

        # MAX ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[1]:
            f.write("** MAX Attention: " + minmax_file_number[1]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(8, 10):
                for y in range(14, 15):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])

            max_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            max_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(max_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(max_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')  

    f.close()


def minmax_target_sample2(PATH_sample2_data_list, drive_name):
    # WARNING: HARD CODE
    file_number = [("00318", "00290")]
    minmax_file_number = file_number[0]

    voxel_feature_min_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    voxel_feature_max_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    f = open(drive_name + '_target.txt', 'w')

    for PATH_data in PATH_sample2_data_list:

        # MIN ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[0]:
            f.write("** MIN Attention: " + minmax_file_number[0]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(10, 12):
                for y in range(14, 15):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])

            min_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            min_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(min_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(min_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')

        # MAX ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[1]:
            f.write("** MAX Attention: " + minmax_file_number[1]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(10, 12):
                for y in range(14, 15):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])

            max_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            max_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(max_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(max_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')  

    f.close()


def minmax_target_sample3(PATH_sample3_data_list, drive_name):
    # WARNING: HARD CODE
    file_number = [("00277", "00283")]
    minmax_file_number = file_number[0]

    voxel_feature_min_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    voxel_feature_max_dict = {
        'num_of_points': list(),
        'x': list(),
        'y': list(),
        'z': list(),
        'dis_to_ego': list(),

        'changes_num_of_points': list(),
        'velocity_to_ego': list(),
        'velocity_x': list(),
        'velocity_y': list(),
        'velocity_z': list(),

        'acceleration_to_ego': list(),
        'acceleration_x': list(),
        'acceleration_y': list(),
        'acceleration_z': list()
    }

    f = open(drive_name + '_target.txt', 'w')

    for PATH_data in PATH_sample3_data_list:

        # MIN ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[0]:
            f.write("** MIN Attention: " + minmax_file_number[0]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(11, 13):
                for y in range(14, 15):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])

            min_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            min_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(min_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(min_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')

        # MAX ATTENTION
        if PATH_data.split('/')[-1].split('\\')[1].split('.')[0] == minmax_file_number[1]:
            f.write("** MAX Attention: " + minmax_file_number[1]+ '\n')

            with open(PATH_data, 'rb') as pk:
                data = pickle.load(pk)

            fm = data['feature_map']
            for x in range(11, 13):
                for y in range(14, 15):
                    for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][x][y])
            for key in voxel_feature_min_dict.keys():
                        voxel_feature_min_dict[key].append(fm[key][11][13])

            max_atten_min_ego = {key:min(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            max_atten_max_ego = {key:max(voxel_feature_min_dict[key]) for key in voxel_feature_min_dict.keys()}
            
            f.write("* MIN feature values\n")
            for key in voxel_feature_min_dict.keys():
                f.write(key+': '+str(max_atten_min_ego[key]))
                f.write('\n')
            f.write('\n')
            f.write("* MAX feature values\n")
            for key in voxel_feature_max_dict.keys():
                f.write(key+': '+str(max_atten_max_ego[key]))
                f.write('\n')
            
            f.write('\n')  

    f.close()

# Path settings
PATH_model = "../../model/model19.pt"
# PATH_sample1_data_list = glob.glob("../../dataset/data_preprocessed/Vulner/21-12-01-11-41-59_end_extract_drive8/*")
# PATH_sample2_data_list = glob.glob("../../dataset/data_preprocessed/Vulner/21-12-14-17-07-28_end_extract_drive12/*")
# PATH_sample3_data_list = glob.glob("../../dataset/data_preprocessed/Vulner/21-12-23-12-13-31_end_extract_drive6/*")

# PATH_sample_data_TN_list = glob.glob("../../dataset/data_preprocessed/None-crash/21-12-01-11-07-44_end_extract_drive3/*") # 300~350
PATH_sample_data_FP_list = glob.glob("../../dataset/data_preprocessed/None-crash/21-12-29-11-04-19_end_extract_drive14/*") # 95~115
# PATH_sample_data_FN_list = glob.glob("../../dataset/data_preprocessed/Vulner/21-12-14-15-04-44_end_extract_drive2/*") # 290~305 -> 292遺��꽣 �엳�쓬

# PATH_sample_data_TN_list = PATH_sample_data_TN_list[297:347+1]
PATH_sample_data_FP_list = PATH_sample_data_FP_list[92:112+1]

# Load model: using CPU
model = torch.load(PATH_model)
model.eval()
model.cpu()

# # Text ego's min & max
# minmax_ego(PATH_sample1_data_list, drive_name="21-12-01-11-41-59_end_extract_drive8")
# minmax_ego(PATH_sample2_data_list, drive_name="21-12-14-17-07-28_end_extract_drive12")
# minmax_ego(PATH_sample3_data_list, drive_name="21-12-23-12-13-31_end_extract_drive6")

# # Text target's min & max
# minmax_target_sample1(PATH_sample1_data_list, drive_name="21-12-01-11-41-59_end_extract_drive8")
# minmax_target_sample2(PATH_sample2_data_list, drive_name="21-12-14-17-07-28_end_extract_drive12")
# minmax_target_sample3(PATH_sample3_data_list, drive_name="21-12-23-12-13-31_end_extract_drive6")

# # Draw Attention mask with points
# for data in PATH_sample1_data_list:
#     vex.draw_img_sample1(model=model, PATH_data=data)

# for data in PATH_sample2_data_list:
#     vex.draw_img_sample2(model=model, PATH_data=data)

# for data in PATH_sample3_data_list:
#     vex.draw_img_sample3(model=model, PATH_data=data)

# for data in PATH_sample_data_FN_list:
#     vex.draw_img_sample_custom(model=model, PATH_data=data)

for data in PATH_sample_data_FP_list:
    vex.draw_img_sample_custom(model=model, PATH_data=data)

# for data in PATH_sample_data_FN_list:
#     vex.draw_img_sample_custom(model=model, PATH_data=data)

# for data in PATH_sample1_data_list:
#     vex.draw_img_lidar_only(model=model, PATH_data=data)


