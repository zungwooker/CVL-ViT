import shutil
import os
import glob
import pickle
import numpy as np

os.makedirs("None-crash_tensor", exist_ok=True)
os.makedirs("Vulner_tensor", exist_ok=True)

None_crash_feature_maps = glob.glob("pickles\\None-crash_feature_map_pickles\\*")
Vulner_feature_maps = glob.glob("pickles\\Vulner_feature_map_pickles\\*")

for fm in None_crash_feature_maps:
    with open(fm, 'rb') as f:
        data = pickle.load(f)

    keys = list(data.keys())[1:]
    tmp_tensor = list()

    for key in keys:
        tmp_tensor.append(data[key])

    tmp_tensor = np.asarray(tmp_tensor)

    result = dict()
    result['label'] = 0
    result['tensor'] = tmp_tensor

    filename = (fm.split('\\')[-1])[:-7]

    with open("None-crash_tensor\\" + filename + "@tensor.pickle", 'wb') as f:
        pickle.dump(result, f)


for fm in Vulner_feature_maps:
    with open(fm, 'rb') as f:
        data = pickle.load(f)

    keys = list(data.keys())[1:]
    tmp_tensor = list()

    for key in keys:
        tmp_tensor.append(data[key])

    tmp_tensor = np.asarray(tmp_tensor)

    result = dict()
    result['label'] = 1
    result['tensor'] = tmp_tensor

    filename = (fm.split('\\')[-1])[:-7]

    with open("Vulner_tensor\\" + filename + "@tensor.pickle", 'wb') as f:
        pickle.dump(result, f)