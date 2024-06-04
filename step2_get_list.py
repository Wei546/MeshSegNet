# numpy用於矩陣運算
import numpy as np
# os用來讀取、寫入文件檔
import os
# sklearn用於機器學習，使用K-Fold交叉驗證
from sklearn.model_selection import KFold
# train_test_split用於將資料分成訓練集和測試集
from sklearn.model_selection import train_test_split
# pandas用於資料處理
import pandas as pd

if __name__ == '__main__':

    #step1_augmentation.py所擴增的資料夾 
    data_path = './augmentation_vtk_data_down/'
    # 把trin_list、val_list、test_list存到當前資料夾
    output_path = './validation_down/'
    # 與step1_augmentation.py的擴增次數相同
    num_augmentations = 20
    # 訓練集佔80%
    train_size = 0.8
    # 確定樣本是否有翻轉
    with_flip = True
    
    # 與step1_augmentation.py的樣本檔案相同
    num_samples = 36 
    sample_list = list(range(1, num_samples+1))
    sample_name = 'A{0}_Sample_0{1}_d.vtp'

    # valid_sample_list的用處為檢察檔案是否重複
    valid_sample_list = []
    for i_sample in sample_list:
        for i_aug in range(num_augmentations):
            if os.path.exists(os.path.join(data_path, sample_name.format(i_aug, i_sample))):
                # 將重複的檔案存入valid_sample_list
                valid_sample_list.append(i_sample)

    # 移除重複的檔案
    sample_list = list(dict.fromkeys(valid_sample_list))
    sample_list = np.asarray(sample_list)
    #print(sample_list)


    # 使用K-Fold交叉驗證
    # K-Fold的意思是，透過資料間的重複採樣過程，用於評估機器學習模型並驗證模型對獨立測試數據集的泛化能力
    i_cv = 0
    # n_splits=6，將資料分成6等分，進行6次驗證
    kf = KFold(n_splits=6, shuffle=False)
    for train_idx, test_idx in kf.split(sample_list):

        # 計算第幾次交叉驗證
        i_cv += 1
        print('Round:', i_cv)

        # 目的為將訓練集，進一步分成訓練集和驗證集
        train_list, test_list = sample_list[train_idx], sample_list[test_idx]
        train_list, val_list = train_test_split(train_list, train_size=0.8, shuffle=True)

        print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

        #產生訓練集(train)csv檔
        train_name_list = []
        # 依照樣本編號、擴增編號，產生新的檔案名稱
        for i_sample in train_list:
            for i_aug in range(num_augmentations):
                #print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
                # 依照樣本編號、擴增編號，產生新的檔案名稱
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                # 將檔案名稱存入train_name_list
                train_name_list.append(os.path.join(data_path, subject_name))
                # 若有翻轉檔案，則將翻轉檔案名稱存入train_name_list
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample+1000)
                    train_name_list.append(os.path.join(data_path, subject2_name))
        # 將train_name_list存入train_list_i_cv.csv
        with open(os.path.join(output_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
            # 將train_name_list的檔案名稱寫入train_list_i_cv.csv
            for f in train_name_list:
                file.write(f+'\n')

        #產生驗證集(val)csv檔
        val_name_list = []
        # 依照樣本編號、擴增編號，產生新的檔案名稱
        for i_sample in val_list:
            for i_aug in range(num_augmentations):
                #print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
                # 依照樣本編號、擴增編號，產生新的檔案名稱
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                # 將檔案名稱存入val_name_list
                val_name_list.append(os.path.join(data_path, subject_name))
                # 若有翻轉檔案，則將翻轉檔案名稱存入val_name_list
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample+1000)
                    val_name_list.append(os.path.join(data_path, subject2_name))
        # 將val_name_list存入val_list_i_cv.csv
        with open(os.path.join(output_path, 'val_list_{0}.csv'.format(i_cv)), 'w') as file:
            # 將val_name_list的檔案名稱寫入val_list_i_cv.csv
            for f in val_name_list:
                file.write(f+'\n')

        #產生測試集(test)csv檔
        test_df = pd.DataFrame(data=test_list, columns=['Test ID'])
        test_df.to_csv('test_list_{}.csv'.format(i_cv), index=False)


        print('--------------------------------------------')
        print('with flipped samples:', with_flip)
        print('# of train:', len(train_name_list))
        print('# of validation:', len(val_name_list))
        print('--------------------------------------------')
