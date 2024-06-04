# os用來讀取、寫入文件檔
import os
# io處理文字、二進位、原始型別之檔案 
import io
# numpy用於矩陣運算
import numpy as np
# vtk用於視覺化、模型處理
import vtk
# vedo用於讀取、寫入、轉換和顯示VTK檔
from vedo import *

# 隨機旋轉、隨機平移和隨機重新縮放，以擴增訓練集
def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)
    return: vtkMatrix4x4
    '''
    # 使用matrix表示vtk
    Trans = vtk.vtkTransform()

    # 旋轉的隨機旗標
    ry_flag = np.random.randint(0,2) 
    rx_flag = np.random.randint(0,2) 
    rz_flag = np.random.randint(0,2) 
    # 隨機旋轉(x、y、z軸)
    if ry_flag == 1:
        # 沿著Y軸旋轉
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # 沿著X軸旋轉
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # 沿著Z軸旋轉
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))
    # 的隨機旗標
    trans_flag = np.random.randint(0,2)
    # 隨機平移
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])
    # 縮放的隨機旗標
    scale_flag = np.random.randint(0,2)
    # 隨機重新縮放
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])
    # 因為tans矩陣封裝了旋轉、平移和縮放，GetMatrix()取得該矩陣
    matrix = Trans.GetMatrix()
    return matrix


if __name__ == "__main__":
    # 丟進去36個標記好的樣本
    num_samples = 36 
    # 標記好的樣本路徑
    # vtk_path = 'C:\\Users\\s6324\\Desktop\\python_file\\MeshSegNet\\all_sample'
    vtk_path = './all_sample/' 
    # 擴增後的檔案路徑
    # output_save_path = 'C:\\Users\\s6324\\Desktop\\python_file\\MeshSegNet\\augmentation_vtk_data'
    output_save_path = './augmentation_vtk_data_down/'
    # 創建擴增檔案的路徑
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)
 
    # 初始化掃描範本列表
    sample_list = list(range(1, num_samples+1))
    # 20組擴增樣本
    num_augmentations = 20
    # 標記範本有36個，共迭帶20組擴增樣本
    for i_sample in sample_list:
        for i_aug in range(num_augmentations):
            # 進行擴增檔案的檔案名稱
            file_name = '{}_down.vtp'.format(i_sample)   
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
            # 圖片預處理
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) 
            # 載入並合併欲輸入vtk檔案+檔名
            mesh = load(os.path.join(vtk_path, file_name))

            mesh.applyTransform(vtk_matrix)
            # 將資料擴增後的vtk檔案存入output_save_path資料夾
            io.write(mesh, os.path.join(output_save_path, output_file_name))

        # 第二層擴增樣本
        for i_aug in range(num_augmentations):
            # 進行擴增檔案的檔案名稱
            file_name = '{}_down.vtp'.format(i_sample+1000)
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample+1000)
            # 圖片預處理
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) 
            mesh = load(os.path.join(vtk_path, file_name))
            mesh.applyTransform(vtk_matrix)
            # 將資料擴增後的vtk檔案存入output_save_path資料夾
            io.write(mesh, os.path.join(output_save_path, output_file_name)) 
