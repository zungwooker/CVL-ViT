import pickle
import numpy as np 
import open3d as o3d
from datetime import datetime
import math
import os
import glob
import copy

# Saved file format
# File name: [source_folder]@[drive_folder]@correction@[pcd_name].pickle
# Content: feature map
# Keys of content(dict): 
#   1. points
#   2. num_of_points
#   3. x
#   4. y
#   5. z
#   6. dis_to_ego
#
#   7. changes_num_of_points
#   8. velocity_to_ego
#   9. velocity_x
#   10. velocity_y
#   11. velocity_z
#
#   12. acceleration_to_ego
#   13. acceleration_x
#   14. acceleration_y
#   15. acceleration_z
# Each map is numpy matrix.

voxel_feature_dict = {
    'points': [],
    'num_of_points': None,
    'x': None,
    'y': None,
    'z': None,
    'dis_to_ego': None,

    'changes_num_of_points': None,
    'velocity_to_ego': None,
    'velocity_x': None,
    'velocity_y': None,
    'velocity_z': None,

    'acceleration_to_ego': None,
    'acceleration_x': None,
    'acceleration_y': None,
    'acceleration_z': None
}

observation_space = {
    'x': 42 * 2,
    'y': 42 * 2,
    'z': 2 * 2
}

voxel_size = {
    'x': 3,
    'y': 3,
    'z': 4
}

num_of_voxel = {
    'x': 28,
    'y': 28,
    'z': 1
}

ego = [observation_space['x']/2, observation_space['y']/2, observation_space['z']/2]

# Determine if pt exists in the observation space
def exist_in_OS(pt):
    # pt is adjusted to the origin coordinates and has only positive values
    x = observation_space['x']
    y = observation_space['y']
    z = observation_space['z']

    if pt[0] > 0 and pt[0] < x:
        if pt[1] > 0 and pt[1] < y:
            if pt[2] > 0 and pt[2] < z:
                return True
    return False


# Returns one feature table(empty frame)
def make_feature_table(feature_format):
    feature_table = [[copy.deepcopy(feature_format) for _ in range(num_of_voxel['y'])] for _ in range(num_of_voxel['x'])]

    return feature_table


# Returns a feature map where feature points are gathered into dict
# keys are feature names
def make_feature_map():
    feature_map = dict()
    features = list(voxel_feature_dict.keys())

    for ft in features:
        feature_map[ft] = make_feature_table(voxel_feature_dict[ft])

    return feature_map


# Inserts points in Voxels corresponding to point coordinates
def points_divider(feature_map, pcd):
    points = np.asarray(o3d.io.read_point_cloud(pcd).points)
    points = points + np.asarray(ego)

    for pt in points:
        if not exist_in_OS(pt):
            continue
        idx_x = int(pt[0]//voxel_size['x'])
        idx_y = int(pt[1]//voxel_size['y'])

        raw_pt = pt - np.asarray(ego)

        feature_map['points'][idx_x][idx_y].append(raw_pt)


# c
def fill_num_of_points(feature_map):
    for x in range(num_of_voxel['x']):
        for y in range(num_of_voxel['y']):
            feature_map['num_of_points'][x][y] = len(feature_map['points'][x][y])

    feature_map['num_of_points'] = np.asarray(feature_map['num_of_points'])


# Fills center coordinates of each voxel
def fill_center(feature_map):
    for x in range(num_of_voxel['x']):
        for y in range(num_of_voxel['y']):
            num_of_points = len(feature_map['points'][x][y])

            if num_of_points:
                center = sum(feature_map['points'][x][y])/len(feature_map['points'][x][y])
            else:
                center = np.asarray([0.0, 0.0, 0.0])

            feature_map['x'][x][y] = center[0]
            feature_map['y'][x][y] = center[1]
            feature_map['z'][x][y] = center[2]

    feature_map['x'] = np.asarray(feature_map['x'])
    feature_map['y'] = np.asarray(feature_map['y'])
    feature_map['z'] = np.asarray(feature_map['z'])


# Fills dis_to_ego feature
def fill_distance(feature_map):
    for x in range(num_of_voxel['x']):
        for y in range(num_of_voxel['y']):
            x_diff = ego[0] - feature_map['x'][x][y]
            y_diff = ego[1] - feature_map['y'][x][y]
            z_diff = ego[2] - feature_map['z'][x][y]

            dis_squared = x_diff**2 + y_diff**2 + z_diff**2
            dis = math.sqrt(dis_squared)

            feature_map['dis_to_ego'][x][y] = dis

    feature_map['dis_to_ego'] = np.asarray(feature_map['dis_to_ego'])



# Extract features that do not require comparison
# 
def feature_map_process0(pcd):
    feature_map = make_feature_map()
    points_divider(feature_map, pcd)
    fill_num_of_points(feature_map)
    fill_center(feature_map)
    fill_distance(feature_map)

    return feature_map

def get_pcd_path_drive(drive):
    PATH_drive = drive + '\\correction\\*.pcd'
    PATH_pcds = glob.glob(PATH_drive)
    return PATH_pcds

# 하나의 드라이브의 pcd들에 대해 process0 실행
def feature_map_process1(drive):
    PATH_pcds = get_pcd_path_drive(drive)
    PATH_pcds.sort()
    
    feature_map_list = list()
    for pcd in PATH_pcds:
        feature_map_dict = dict()
        feature_map_dict['path'] = pcd.replace('\\', '@')
        feature_map_dict['feature_map'] = feature_map_process0(pcd)
        feature_map_list.append(feature_map_dict)

    return feature_map_list

# 하나의 드라이브 pcd들에 대해 1차 변화량 특징점 채우기
def feature_map_process2(feature_map_list):
    # Changes num of points
    for i in range(1, len(feature_map_list)):
        changes_num_of_points = feature_map_list[i]['feature_map']['num_of_points'] - feature_map_list[i-1]['feature_map']['num_of_points']

        feature_map_list[i]['feature_map']['changes_num_of_points'] = changes_num_of_points

    # velocity to ego
    for i in range(1, len(feature_map_list)):
        velocity_to_ego = feature_map_list[i]['feature_map']['dis_to_ego'] - feature_map_list[i-1]['feature_map']['dis_to_ego']

        feature_map_list[i]['feature_map']['velocity_to_ego'] = velocity_to_ego

    # velocity x
    for i in range(1, len(feature_map_list)):
        velocity_x = feature_map_list[i]['feature_map']['x'] - feature_map_list[i-1]['feature_map']['x']

        feature_map_list[i]['feature_map']['velocity_x'] = velocity_x

    # velocity y
    for i in range(1, len(feature_map_list)):
        velocity_y = feature_map_list[i]['feature_map']['y'] - feature_map_list[i-1]['feature_map']['y']

        feature_map_list[i]['feature_map']['velocity_y'] = velocity_y

    # velocity z
    for i in range(1, len(feature_map_list)):
        velocity_z = feature_map_list[i]['feature_map']['z'] - feature_map_list[i-1]['feature_map']['z']

        feature_map_list[i]['feature_map']['velocity_z'] = velocity_z


# 하나의 드라이브 pcd들에 대해 2차 변화량 특징점 채우기
def feature_map_process3(feature_map_list):
    # acceleration to ego
    for i in range(2, len(feature_map_list)):
        acceleration_to_ego = feature_map_list[i]['feature_map']['velocity_to_ego'] - feature_map_list[i-1]['feature_map']['velocity_to_ego']

        feature_map_list[i]['feature_map']['acceleration_to_ego'] = acceleration_to_ego

    # acceleration x
    for i in range(2, len(feature_map_list)):
        acceleration_x = feature_map_list[i]['feature_map']['velocity_x'] - feature_map_list[i-1]['feature_map']['velocity_x']

        feature_map_list[i]['feature_map']['acceleration_x'] = acceleration_x

    # acceleration y
    for i in range(2, len(feature_map_list)):
        acceleration_y = feature_map_list[i]['feature_map']['velocity_y'] - feature_map_list[i-1]['feature_map']['velocity_y']

        feature_map_list[i]['feature_map']['acceleration_y'] = acceleration_y

    # acceleration z
    for i in range(2, len(feature_map_list)):
        acceleration_z = feature_map_list[i]['feature_map']['velocity_z'] - feature_map_list[i-1]['feature_map']['velocity_z']

        feature_map_list[i]['feature_map']['acceleration_z'] = acceleration_z


def make_pickles_drive(drive, save_folder):
    # drive: None-crash_unzip\\21-12-01-11-07-44_end_extract_drive14.zip(폴더명)
    # path: None-crash_unzip\\21-12-01-11-07-44_end_extract_drive14.zip\\correction\\00001.pcd

    feature_map_list = feature_map_process1(drive)
    feature_map_process2(feature_map_list)
    feature_map_process3(feature_map_list)
    feature_map_list = feature_map_list[2:] # 불완전한 전방 2장(0, 1) 제거

    drive_path = drive.split("\\")

    if drive_path[0] == 'None-crash_unzip':
        folder = 'None-crash_feature_map_pickles'
    else:
        folder = 'Vulner_feature_map_pickles'

    save_path = save_folder + "\\" + folder + "\\" + drive_path[1]
    # save folder 안에 folder들을 미리 만들어 놔야함
    os.mkdir(save_path)

    for fm in feature_map_list:
        with open(save_path + "\\" + fm['path'] + ".pickle", "wb") as f:
            pickle.dump(fm['feature_map'], f, pickle.HIGHEST_PROTOCOL)

def drive_list(folder):
    drives = glob.glob(folder + "\\*")
    return drives

def extract_all(save_folder):
    vulner_drives = drive_list("Vulner_unzip")
    none_crash_drives = drive_list("None-crash_unzip")

    for drive in vulner_drives:
        print("Extracting...")
        print(drive)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '\n')
        make_pickles_drive(drive, save_folder)

    for drive in none_crash_drives:
        print("Extracting...")
        print(drive)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '\n')
        make_pickles_drive(drive, save_folder)

def main():
    save_folder = "pickles"
    os.mkdir(save_folder + "\\None-crash_feature_map_pickles")
    os.mkdir(save_folder + "\\Vulner_feature_map_pickles")

    extract_all(save_folder)

main()