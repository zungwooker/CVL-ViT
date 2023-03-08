import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import pickle

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

# Path settings
PATH_model = "../../model/model.pt"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


def show_mask(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def match_mask_to_voxel(mask, patch_size):
    # Unit of return mask is voxel
    mask_size = mask.shape 
    mask_voxel_matched = np.zeros((mask_size[0]*patch_size, mask_size[1]*patch_size))

    for x in range(mask_size[0]):
        for y in range(mask_size[1]):
            for x_window in range(patch_size):
                for y_window in range(patch_size):
                    mask_voxel_matched[patch_size*x + x_window][patch_size*y + y_window] = mask[x][y]

    return mask_voxel_matched


def save_points_with_attention(mask_matched_to_voxel, input_points, name):
    plt.figure(figsize=(24,20))

    plt.xticks(np.arange(-42,42,3))
    plt.yticks(np.arange(-42,42,3))
    
    plt.xlabel('Y-Axis')
    plt.ylabel('X-Axis')
    
    plt.axis([-42,42,-42,42])

    plt.grid(linestyle=':', linewidth=1)

    x_coor = list()
    y_coor = list()
    for x in range(len(input_points)):
        for y in range(len(input_points[x])):
            for pt in input_points[x][y]:
                x_coor.append(pt[0])
                y_coor.append(pt[1])

    plt.scatter(y_coor, x_coor, marker='o', s=0.05)

    for x in range(28):
        for y in range(28):
            plt.fill(\
                [-42 + (y+0)*3, -42 + (y+1)*3, -42 + (y+1)*3, -42 + (y+0)*3], \
                [-42 + (x+0)*3, -42 + (x+0)*3, -42 + (x+1)*3, -42 + (x+1)*3], \
                color='red', alpha=mask_matched_to_voxel[x][y]*0.9)
    
    plt.gca().invert_yaxis()

    # colormap = plt.cm.get_cmap('plasma')
    colormap = plt.cm.get_cmap('Reds')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    plt.colorbar(sm)
    plt.savefig(name + '.png')
    plt.close()


def save_points_with_attention_and_number(mask_matched_to_voxel, input_points, name):
    plt.figure(figsize=(24,20))

    plt.xticks(np.arange(-42,42,3))
    plt.yticks(np.arange(-42,42,3))
    
    plt.xlabel('Y-Axis')
    plt.ylabel('X-Axis')
    
    plt.axis([-42,42,-42,42])

    plt.grid(linestyle=':', linewidth=1)

    for x in np.arange(-42,42,6):
        for y in np.arange(-42,42,6):
            plt.text(x=y+1.5, y=x+3, s='('+str((x+42)//6)+', '+str((y+42)//6)+')')

    x_coor = list()
    y_coor = list()
    for x in range(len(input_points)):
        for y in range(len(input_points[x])):
            for pt in input_points[x][y]:
                x_coor.append(pt[0])
                y_coor.append(pt[1])

    plt.scatter(y_coor, x_coor, marker='o', s=0.05)

    for x in range(28):
        for y in range(28):
            plt.fill(\
                [-42 + (y+0)*3, -42 + (y+1)*3, -42 + (y+1)*3, -42 + (y+0)*3], \
                [-42 + (x+0)*3, -42 + (x+0)*3, -42 + (x+1)*3, -42 + (x+1)*3], \
                color='red', alpha=mask_matched_to_voxel[x][y]*0.9)
    
    plt.gca().invert_yaxis()

    # colormap = plt.cm.get_cmap('plasma')
    colormap = plt.cm.get_cmap('Reds')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    plt.colorbar(sm)
    plt.savefig(name + '.png')
    plt.close()


def save_points_only(input_points, name, PATH_save):
    plt.figure(figsize=(24,20))

    plt.xticks(np.arange(-42,42,3))
    plt.yticks(np.arange(-42,42,3))
    
    plt.xlabel('Y-Axis')
    plt.ylabel('X-Axis')
    
    plt.axis([-42,42,-42,42])

    plt.grid(linestyle=':', linewidth=1)

    x_coor = list()
    y_coor = list()
    for x in range(len(input_points)):
        for y in range(len(input_points[x])):
            for pt in input_points[x][y]:
                x_coor.append(pt[0])
                y_coor.append(pt[1])

    plt.scatter(y_coor, x_coor, marker='o', s=0.05)
    
    plt.gca().invert_yaxis()
    plt.savefig(PATH_save + '/' + name + '.png')
    plt.close()


def draw_img(model, PATH_data):
    # Load data: using CPU
    with open(PATH_data, 'rb') as f:
        data = pickle.load(f)

    input_tensor = data['tensor']
    input_points = data['feature_map']['points']

    input_tensor = input_tensor.reshape((1,14,28,28))
    input_tensor = input_tensor.to(torch.float32).cpu()

    # Rollout attentions
    attention_rollout = VITAttentionRollout(model, head_fusion='max', discard_ratio=0.9)
    mask_law = attention_rollout(input_tensor) # (14, 14)
    
    # Match mask to OS
    mask_matched_to_voxel = match_mask_to_voxel(mask=mask_law, patch_size=2)

    # name image(ex. 21-12-01-11-41-59_end_extract_drive8_00287)
    name = PATH_data.split('/')[-1].split('\\')[0] + '_' + PATH_data.split('/')[-1].split('\\')[1].split('.')[0]

    # Draw pcd and attentions and save it
    save_points_with_attention(mask_matched_to_voxel, input_points, name)


def draw_img_lidar_only(PATH_data, PATH_save):
    # Load data: using CPU
    with open(PATH_data, 'rb') as f:
        data = pickle.load(f)

    input_points = data['feature_map']['points']

    # name image(ex. 21-12-01-11-41-59_end_extract_drive8_00287)
    name = PATH_data.split('/')[-1].split('\\')[0] + '_' + PATH_data.split('/')[-1].split('\\')[1].split('.')[0] + '_lidar'

    # Draw pcd and attentions and save it
    save_points_only(input_points, name, PATH_save)


def draw_img_sample_custom(model, PATH_data, label):
    # name image(ex. 21-12-01-11-41-59_end_extract_drive8_00287)
    name = PATH_data.split('/')[-1].split('\\')[0] + '_' + PATH_data.split('/')[-1].split('\\')[1].split('.')[0]

    with open(PATH_data, 'rb') as f:
        data = pickle.load(f)

    input_tensor = data['tensor']
    input_points = data['feature_map']['points']

    input_tensor = input_tensor.reshape((1,14,28,28))
    input_tensor = input_tensor.to(torch.float32).cpu()

    # Rollout attentions
    attention_rollout = VITAttentionRollout(model, head_fusion='max', discard_ratio=0.9)
    mask_law = attention_rollout(input_tensor) # (14, 14)

    if label == 'TP':
        # Edit attentions
        for x in range(0, 13+1): # 차선 밖
            for y in range(0, 3+1):
                mask_law[x][y] = mask_law[x][y] * 0.5
            for y in range(9, 13+1):
                mask_law[x][y] = mask_law[x][y] * 0.5
        
        for x in range(0, 13+1): # 반대편 차선 가드레일
            mask_law[x][4] = mask_law[x][4] * 0.35
        
        for x in range(0, 13+1): # 반대편 차선
            mask_law[x][5] = mask_law[x][5] * 0.5

        for x in range(8, 13+1): # Ego Vehicle 뒤
            for y in range(6, 8+1):
                mask_law[x][y] = mask_law[x][y] * 0.5

        # 특이 지점
        mask_law[6][7] = mask_law[6][7] * 0.3
        mask_law[7][6] = mask_law[7][6] * 0.3
        mask_law[8][6] = mask_law[8][6] * 0.3
        mask_law[9][6] = mask_law[9][6] * 0.3

    elif label == 'TN':
        # Edit attention
        for x in range(0, 13+1): # 반대편 차선
            for y in range(0, 4+1):
                mask_law[x][y] = mask_law[x][y] * 0.5
        
        for x in range(0, 13+1): # 중앙분리대
            mask_law[x][5] = mask_law[x][5] * 0.3

        mask_law[9][7] = mask_law[9][7] * 0.05 # 특이지점

        for x in range(7, 13+1): # Ego vehicle 뒤 차도
            for y in range(6, 8+1):
                mask_law[x][y] = mask_law[x][y] * 0.3

        for x in range(0, 13+1): # 차도 밖
            for y in range(9, 13+1):
                mask_law[x][y] = mask_law[x][y] * 0.6

    elif label == 'FP':
        # Edit attention
        mask_law[5][4] = mask_law[5][4] * 0.2 # 특이지점
        
    elif label == 'FN':
        # Edit attention
        for x in range(0, 13+1): # 차도밖
            for y in range(0, 3+1):
                mask_law[x][y] = mask_law[x][y] * 0.15 
            for y in range(9, 13+1):
                mask_law[x][y] = mask_law[x][y] * 0.15

        for x in range(0, 13+1): # 반대편 차선
            for y in range(4, 5+1):
                mask_law[x][y] = mask_law[x][y] * 0.2
        
        for x in range(8, 13+1): #Ego Vehicle 뒤
            for y in range(6, 7+1):
                mask_law[x][y] = mask_law[x][y] * 0.5
    
    elif label == 'NO MASKING':
        pass
    
    
    # Match mask to OS
    mask_matched_to_voxel = match_mask_to_voxel(mask=mask_law, patch_size=2)

    # Draw pcd and attentions and save it
    save_points_with_attention_and_number(mask_matched_to_voxel, input_points, name)