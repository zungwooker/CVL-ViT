from zipfile import ZipFile
import pandas as pd
import os
import datetime
import shutil

def unzip_None_crash(PATH_None_crash):
    print("========== Unzipping None-crash ==========", '\n')
    
    PATH_None_crash_unzip = '../../dataset/data_unzip/None-crash'
    os.makedirs(PATH_None_crash_unzip, exist_ok = True)

    # None-crash 폴더의 zip 파일 이름 리스트 생성
    drives = os.listdir(PATH_None_crash)
    for i in range(len(drives)):
        drives[i] = drives[i].split('.')[0] # elements without filename extension

    for drive in drives:
        PATH_zip_folder = PATH_None_crash + '/' + drive + '.zip'
        PATH_unzip_folder = PATH_None_crash_unzip + '/' + drive

        with ZipFile(PATH_zip_folder, 'r') as zipObj:
            zipObj.extractall(PATH_unzip_folder)

        # move files out of '/correction'
        pcds = os.listdir(PATH_unzip_folder + '/correction')
        for pcd in pcds:
            PATH_pcd = PATH_unzip_folder + '/correction/' + pcd
            shutil.move(PATH_pcd, PATH_unzip_folder)
        shutil.rmtree(PATH_unzip_folder + '/correction')

        print("Unzipped: " + drive)
        print(datetime.datetime.now(), '\n')

    print("======== All None-crash upzipped =========", '\n')


def unzip_Vulner(PATH_excel, PATH_Vulner):
    print("=========== Unzipping Vulner ===========", '\n')

    PATH_Vulner_unzip = '../../dataset/data_unzip/Vulner'
    os.makedirs(PATH_Vulner_unzip, exist_ok = True)

    # DataFrame 형태로 엑셀에서 필요한 columns 불러오기
    # column name: 13='폴더_Name', 14='drive_Num', 21='움찔 추정 시간', 28='학습용구분', 29='LiDAR 추출'
    df = pd.read_excel( 
                    PATH_excel, 
                    sheet_name= 'TS_HDD_03', 
                    usecols= [13, 14, 21, 28, 29]
                    )
    
    df.to_csv('tmp.csv')
    exit()

    # 조건을 만족하는 행만 추출
    # 1. 학습용 구분: Vulnerable
    # 2. LiDAR 추출: 0
    # 3. 움찔 추정 시간: not x
    is_vulner = df['학습용구분'] == 'Vulnerable'
    is_ok = df['LiDAR 추출'] == 0
    is_flinch = df['움찔 추정 시간'] != 'x'
    subset_df = df[is_vulner & is_ok & is_flinch]

    # 엑셀 파일 내의 drive 이름 추출
    drives = list()
    for folder, drive in zip(subset_df['폴더_Name'], subset_df['drive_Num']):
        name = folder + '_drive' + drive[6:]
        drives.append(name)

    # 컬럼명 drive의 zip파일 이름 형태의 column 추가
    subset_df.insert(loc=0, column='drive', value=drives)

    #Vulner 폴더에서 drive과 같은 파일 찾기
    for drive, time in zip(subset_df['drive'], subset_df['움찔 추정 시간']):
        PATH_zip_folder = PATH_Vulner + '/' + drive + '.zip'
        PATH_unzip_folder = PATH_Vulner_unzip + '/' + drive

        if os.path.isfile(PATH_Vulner + "/" + drive + ".zip"):

            #움찔 추정 기간 가져오기(start, end)
            start = int(time[:3])
            end = int(time[4:])

            with ZipFile(PATH_zip_folder, 'r') as zipObj:
                pcd_names = zipObj.namelist()

                #start부터 end 까지만 추출 후 압축 풀어 저장
                for pcd_num in range(start, end+1):
                    for pcd_name in pcd_names:
                        if str(pcd_num) in pcd_name:
                            zipObj.extract(pcd_name, PATH_unzip_folder)

            # move files out of '/correction'
            pcds = os.listdir(PATH_unzip_folder + '/correction')
            for pcd in pcds:
                PATH_pcd = PATH_unzip_folder + '/correction/' + pcd
                shutil.move(PATH_pcd, PATH_unzip_folder)
            shutil.rmtree(PATH_unzip_folder + '/correction')

        print("Unzipped: " + drive)
        print(datetime.datetime.now(), '\n')

    print("========= All Vulner upzipped ==========", '\n')
                            

def main():
    PATH_excel = '../../dataset/TS_HDD_03_Lidar_ViT.xlsx'
    PATH_None_crash = "../../dataset/data_law/None-crash"
    PATH_Vulner = "../../dataset/data_law/Vulner"

    os.makedirs('../../dataset/data_unzip', exist_ok = True)

    unzip_Vulner(PATH_excel, PATH_Vulner)
    unzip_None_crash(PATH_None_crash)


if __name__ == "__main__":
    main()
