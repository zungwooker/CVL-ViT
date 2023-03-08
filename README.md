# README

**Vulnerable Voxel Space(VVS) and Vision Transformer(ViT)**    
Mar. 2022 ~ Present

## Structure
```
📂 VVS-ViT/
├── 📂 dataset/
│   ├── 📂 data_law/
│   |	├── 📂 None-crash/
│   |	|	├── 📄 21-01-01-11-13-10_end_extract_drive1.zip
│   |	|	├── 📄 21-01-01-11-13-10_end_extract_drive14.zip
│   |	|	└── 📄 ...
│   |	|	
│   |	└── 📂 Vulner/
│   |		├── 📄 21-01-01-11-50-10_end_extract_drive8.zip
│   |		├── 📄 21-01-01-11-54-15_end_extract_drive2.zip
│   |		└── 📄 ...
│   |
│   ├── 📂 data_preprocessed/
│   |	├── 📂 None-crash/
│   |	|	├── 📂 21-01-01-11-13-10_end_extract_drive1/
│   |	|	|	├── 📄 00003.pickle
│   |	|	|	├── 📄 00004.pickle
│   |	|	|	├── 📄 ...
│   |	|	|	└── 📄 00600.pickle
│   |	|	|
│   |	|	└── 📂 ...
│   |	|
│   |	└── 📂 Vulner/
│   |		├── 📂 21-01-01-11-50-10_end_extract_drive8/
│   |		|	├── 📄 00287.pickle
│   |		|	├── 📄 ...
│   |		|	└── 📄 00325.pickle
│   |		|
│   |		├── 📂 21-01-01-11-54-15_end_extract_drive2/
│   |		|	├── 📄 00102.pickle
│   |		|	├── 📄 ...
│   |		|	└── 📄 00198.pickle
│   |		|
│   |		└── 📂 ...
│   |
│   ├── 📂 data_unzip/
│   |	├── 📂 None-crash/
│   |	|	├── 📂 21-01-01-11-13-10_end_extract_drive1/
│   |	|	|	├── 📄 00001.pcd
│   |	|	|	├── 📄 00002.pcd
│   |	|	|	├── 📄 ...
│   |	|	|	└── 📄 00600.pcd
│   |	|	|
│   |	|	└── 📂 ...
│   |	|
│   |	└── 📂 Vulner/
│   |		├── 📂 21-01-01-11-50-10_end_extract_drive8/
│   |		|	├── 📄 00285.pcd
│   |		|	├── 📄 ...
│   |		|	└── 📄 00325.pickle
│   |		|
│   |		├── 📂 21-01-01-11-54-15_end_extract_drive2/
│   |		|	├── 📄 00100.pickle
│   |		|	├── 📄 ...
│   |		|	└── 📄 00198.pickle
│   |		|
│   |		└── 📂 ...
│   |
│   └── 📄 TS_HDD_03_Lidar_ViT.xlsx
|
├── 📂 model/
│   ├── 🤖 model.pt
│   └── ...
│
├── 📂 scripts/
│   ├── 📂 attention_rollout/
│   ├── 📂 model_train/
│   ├── 📂 pcd_preprocessor/
│   ├── 📂 unzipers/
│   └── ...
│
└── ...
```
* All codes are in `scripts` folder.
* Data should be prepared before run the codes.
* We do not provide dataset.

## Required Libraries

Writing...

## Usage

1. Place PCD zip files in `dataset/data_law/None-crash` and `dataset/data_law/Vulner`. Drives are should be separated by labels.
2. Prepare `TS_HDD_03_Lidar_ViT.xlsx` file with pcd information organized and place it in `dataset/`
3. Run `unzipers/unzip_drives.py`. It unzips your zipped law data based on `TS_HDD_03_Lidar_ViT.xlsx`. 
	> `unzip_drives.py` optionally unzips data from vulnerable drives because not all the PCD files are vulnerable situation in the drives.
4. Run `pcd_preprocessor/lidar_extractor{latest version}.py`. It preprocesses PCD files to pickle files for training VVS-ViT.
5. Run `model_train/data_split.ipynb`. It splits dataset into `train`, `valid`, `test` and save the files path as `.pickle`.
6. Run `model_train/vit_tune.ipynb` to train a new model and save it. You can also check out confusion matrix with test dataset.
7. Now you are ready for attention rollout. Run the `.ipynb` in `attention_rollout/` according to the required operation.

## Reference
1. https://github.com/lucidrains/vit-pytorch
2. https://github.com/jacobgil/vit-explain