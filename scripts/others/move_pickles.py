import shutil
import os
import glob
import pickle

# pickles\None-crash_feature_map_pickles\21-12-01-11-07-44_end_extract_drive14.zip\None-crash_unzip@21-12-01-11-07-44_end_extract_drive14.zip@correction@00003.pcd.pickle

drives = glob.glob("pickles\\Vulner_feature_map_pickles\\*")
drives.sort()

for drive in drives:
    print(drive)
    files = glob.glob(drive+"\\*")
    for file in files:
        filepath = file.split('\\')

        src = file
        dir = filepath[0] + '\\' + filepath[1] + '\\' + filepath[3]

        shutil.move(src, dir)
