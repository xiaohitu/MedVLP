"""
file: PPMi_3D_dataset
author: 
create time: 2024/7/3
last modified: 2024/7/3/11:51
version: 1.0
description: 
"""

import random
import os
import numpy as np
import os
import random
import numpy as np
import torch.utils.data as data_utils
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk
import torch

from tqdm import tqdm, trange
import monai.transforms as mtf

def load_Data2Txt(origin_path=r"D:\data\PPMI\PPMI_nii_align_data", txt_path='./', classifier=["Control", "PD"]):
    # 保证可复现
    # random.seed(1)
    data_path = []
    for label, cls in enumerate(classifier):
        subject_path = os.path.join(origin_path, cls)
        for file in os.listdir(subject_path):
            path = os.path.join(subject_path, file)
            data_path.append((path, label))

    # 使用 with open 写入 train_list.txt
    with open(os.path.join(txt_path, 'data_list.txt'), 'w') as train_txt:
        for item in data_path:
            # 假设元组中的元素都是字符串，可以直接连接
            line = item[0] + ' ' + str(item[1]) + '\n'
            train_txt.write(line)

    print("dataset load over! The files are saved in " + txt_path + "data_list.txt")


def load_Data2Txt3cls(origin_path=r"D:\data\PPMI\PPMI_nii_align_data", txt_path='./',
                      classifier=["Control", "Prodromal", "PD"]):
    # 保证可复现
    # random.seed(1)
    data_path = []
    for label, cls in enumerate(classifier):
        subject_path = os.path.join(origin_path, cls)
        for file in os.listdir(subject_path):
            path = os.path.join(subject_path, file)
            data_path.append((path, label))

    # 使用 with open 写入 train_list.txt
    with open(os.path.join(txt_path, 'data_list.txt'), 'w') as train_txt:
        for item in data_path:
            # 假设元组中的元素都是字符串，可以直接连接
            line = item[0] + ' ' + str(item[1]) + '\n'
            train_txt.write(line)

    print("Dataset load over! The files are saved in " + txt_path + "data_list.txt")


def loadTxt2List(file_path):
    '''
    :param file_path:存放nii数据路径的txt文件
    :return:
    '''
    image_paths = []
    labels = []

    # 打开文件并按行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符
            line = line.strip()
            # 分割每一行以获取图像路径和标签
            parts = line.split()
            # 假设路径和标签之间至少有一个空格分隔
            if len(parts) >= 2:
                image_path = parts[0]
                label = parts[1]
                # 将图像路径和标签分别添加到列表中
                image_paths.append(image_path)
                labels.append(int(label))
    return image_paths, labels


def loadDataByCls(path_origin=r"D:\data\PPMI\PPMI_nii_align_data", txt_path='./', clsname='PD'):
    classifier = {"Control": 0, "PD": 1}
    data_path = []
    subject_path = os.path.join(path_origin, clsname)
    for file in os.listdir(subject_path):
        path = os.path.join(subject_path, file)
        data_path.append((path, classifier[clsname]))

    # 使用 with open 写入 list.txt
    with open(os.path.join(txt_path, clsname + '_data_list.txt'), 'w') as train_txt:
        for item in data_path:
            # 假设元组中的元素都是字符串，可以直接连接
            line = item[0] + ' ' + str(item[1]) + '\n'
            train_txt.write(line)

    print(clsname + "dataset split over! The files are saved in " + str(txt_path))



# 先利用xjh的增强方法，然后在利用Monai的增强方法
class PPMI_dataset_withAugMonai(data_utils.Dataset):
    def __init__(self, data_path_list, label_list, is_training=False, transforms=None):
        super().__init__()
        self.data_path_list = data_path_list
        self.label_list = label_list
        self.is_training = is_training
        self.transforms = transforms
        self.data, self.labels = self._load_data()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _read_Nifit(self, img_path):
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        if self.is_training == True:
            rand1 = torch.rand(1)[0]
            if rand1 >= 0 and rand1 < 0.166:
                # print('Option 1')
                img_array = img_array[0:140, 11:181, 10:150]
            elif rand1 >= 0.166 and rand1 < 0.333:
                img_array = img_array[2:142, 12:182, 10:150]
                # print('Option 2')
            elif rand1 >= 0.333 and rand1 < 0.499:
                img_array = img_array[1:141, 11:181, 10:150]
                # print('Option 3')
            elif rand1 >= 0.499 and rand1 < 0.666:
                img_array = img_array[1:141, 13:183, 10:150]
                # print('Option 4')
            elif rand1 >= 0.666 and rand1 < 0.833:
                img_array = img_array[1:141, 12:182, 9:149]
                # print('Option 5')
            else:
                img_array = img_array[1:141, 12:182, 11:151]
                # print('Option 6')
        else:
            # Test phase
            img_array = img_array[1:141, 12:182, 10:150]

        # Mirror left and right brain
        if self.is_training == True:
            rand = torch.rand(1)[0]
            if rand > 0.5:
                img_array = img_array[:, :, ::-1]
        # img_array = img_array / img_array.max()
        img = np.expand_dims(img_array, axis=0)
        # imgsize=(160,192,160)
        # target_size=(32,256,256)
        if self.transforms:
            img = self.transforms(img)

        return img

    def _load_data(self):
        data = []
        labels = []
        text ="Training" if self.is_training else "Validation"
        for path, label in tqdm(zip(self.data_path_list, self.label_list), desc=f'Loading {text} data',
                                total=len(self.data_path_list)):
            img = self._read_Nifit(path)
            img = np.array(img, dtype='float32')
            data.append(img)
            labels.append(np.array(label))
        return data, labels


class PPMI_T1dataset_withAugMonai(data_utils.Dataset):
    def __init__(self, data_path_list, label_list, is_training=False, transforms=None):
        '''
        AugMonai是在Monai中使用的基础之上增加了xjh的随机裁剪的方法
        :param data_path_list:
        :param label_list:
        :param is_training:
        :param transforms:
        '''
        super().__init__()
        self.data_path_list = data_path_list
        self.label_list = label_list
        self.is_training = is_training
        self.transforms = transforms
        self.data, self.labels = self._load_data()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _read_Nifit(self, img_path):
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        # if self.is_training == True:
        #     rand1 = torch.rand(1)[0]
        #     if rand1 >= 0 and rand1 < 0.166:
        #         # print('Option 1')
        #         img_array = img_array[0:160, 13:203, 10:170]
        #     elif rand1 >= 0.166 and rand1 < 0.333:
        #         img_array = img_array[2:162, 13:203, 10:170]
        #         # print('Option 2')
        #     elif rand1 >= 0.333 and rand1 < 0.499:
        #         img_array = img_array[1:161, 12:202, 10:170]
        #         # print('Option 3')
        #     elif rand1 >= 0.499 and rand1 < 0.666:
        #         img_array = img_array[1:161, 14:204, 10:170]
        #         # print('Option 4')
        #     elif rand1 >= 0.666 and rand1 < 0.833:
        #         img_array = img_array[1:161, 13:203, 9:169]
        #         # print('Option 5')
        #     else:
        #         img_array = img_array[1:161, 13:203, 11:171]
        #         # print('Option 6')
        # else:
        #     # Test phase
        #     img_array = img_array[1:161, 13:203, 10:170]
        if self.is_training == True:
            rand1 = torch.rand(1)[0]
            if rand1 >= 0 and rand1 < 0.166:
                # print('Option 1')
                img_array = img_array[0:160, 12:204, 10:170]
            elif rand1 >= 0.166 and rand1 < 0.333:
                img_array = img_array[2:162, 12:204, 10:170]
                # print('Option 2')
            elif rand1 >= 0.333 and rand1 < 0.499:
                img_array = img_array[1:161, 11:203, 10:170]
                # print('Option 3')
            elif rand1 >= 0.499 and rand1 < 0.666:
                img_array = img_array[1:161, 13:205, 10:170]
                # print('Option 4')
            elif rand1 >= 0.666 and rand1 < 0.833:
                img_array = img_array[1:161, 12:204, 9:169]
                # print('Option 5')
            else:
                img_array = img_array[1:161, 12:204, 11:171]
                # print('Option 6')
        else:
            # Test phase
            img_array = img_array[1:161, 12:204, 10:170]
        # Mirror left and right brain
        if self.is_training == True:
            rand = torch.rand(1)[0]
            if rand > 0.5:
                img_array = img_array[:, :, ::-1]

        #img=(D,H,W)=(160,192,160)
        img = np.expand_dims(img_array, axis=0)
        # img=(1,D,H,W)=(1,160,192,160)
        # if self.transforms:
        #     img = self.transforms(img)
        # resize_transform = mtf.Resize((32, 256, 256))
        # img = resize_transform(img)
        #img=(1,32,256,256)
        #归一化
        img = img / img.max()

        return img

    def _load_data(self):
        data = []
        labels = []
        text ="Training" if self.is_training else "Validation"
        for path, label in tqdm(zip(self.data_path_list, self.label_list), desc=f'Loading {text} data',
                                total=len(self.data_path_list)):
            img = self._read_Nifit(path)
            img = np.array(img, dtype='float32')
            data.append(img)
            labels.append(np.array(label))
        return data, labels

# Res+M3D 组合使用的数据加载方式
class PPMI_SAMdataset_withAugMonai(data_utils.Dataset):
    def __init__(self, data_path_list, label_list, is_training=False, transforms=None):
        '''
        AugMonai是在Monai中使用的基础之上增加了xjh的随机裁剪的方法
        :param data_path_list:
        :param label_list:
        :param is_training:
        :param transforms:
        '''
        super().__init__()
        self.data_path_list = data_path_list
        self.label_list = label_list
        self.is_training = is_training
        self.transforms = transforms
        self.data, self.labels = self._load_data()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _read_Nifit(self, img_path):
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        if self.is_training == True:
            rand1 = torch.rand(1)[0]
            if rand1 >= 0 and rand1 < 0.166:
                # print('Option 1')
                img_array = img_array[0:160, 12:202, 10:170]
            elif rand1 >= 0.166 and rand1 < 0.333:
                img_array = img_array[2:162, 12:202, 10:170]
                # print('Option 2')
            elif rand1 >= 0.333 and rand1 < 0.499:
                img_array = img_array[1:161, 12:202, 10:170]
                # print('Option 3')
            elif rand1 >= 0.499 and rand1 < 0.666:
                img_array = img_array[1:161, 14:204, 10:170]
                # print('Option 4')
            elif rand1 >= 0.666 and rand1 < 0.833:
                img_array = img_array[1:161, 13:203, 9:169]
                # print('Option 5')
            else:
                img_array = img_array[1:161, 13:203, 11:171]
                # print('Option 6')
        else:
            # Test phase
            img_array = img_array[1:161, 13:203, 10:170]

        # Mirror left and right brain
        if self.is_training == True:
            rand = torch.rand(1)[0]
            if rand > 0.5:
                img_array = img_array[:, :, ::-1]

        #img=(D,H,W)=(160,192,160)
        img = np.expand_dims(img_array, axis=0)
        # img=(1,D,H,W)=(1,160,192,160)
        # if self.transforms:
        #     img = self.transforms(img)
        resize_transform = mtf.Resize((80, 96, 80))
        img = resize_transform(img)
        #img=(1,32,256,256)
        #归一化
        img = img / img.max()

        return img

    def _load_data(self):
        data = []
        labels = []
        text ="Training" if self.is_training else "Validation"
        for path, label in tqdm(zip(self.data_path_list, self.label_list), desc=f'Loading {text} data',
                                total=len(self.data_path_list)):
            img = self._read_Nifit(path)
            img = np.array(img, dtype='float32')
            data.append(img)
            labels.append(np.array(label))
        return data, labels


class PPMI_dataset_with_Monai(data_utils.Dataset):
    def __init__(self, data_path_list, label_list, is_training=False, transforms=None):
        super().__init__()
        self.data_path_list = data_path_list
        self.label_list = label_list
        self.is_training = is_training
        self.transforms = transforms
        self.data, self.labels = self._load_data()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _read_Nifit(self, img_path):
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        img = np.expand_dims(img, axis=0)
        # imgsize=(160,192,160)
        # target_size=(32,256,256)
        if self.transforms:
            img = self.transforms(img)

        return img

    def _load_data(self):
        data = []
        labels = []
        text ="Training" if self.is_training else "Validation"
        for path, label in tqdm(zip(self.data_path_list, self.label_list), desc=f'Loading {text} data',
                                total=len(self.data_path_list)):
            img = self._read_Nifit(path)
            img = np.array(img, dtype='float32')
            data.append(img)
            labels.append(np.array(label))
        return data, labels



if __name__ == "__main__":
    origin_path = r"D:\data\PPMI\T1_cropAlign"
    txt_path = "./data_list.txt"
    # load_Data2Txt(origin_path=origin_path)
    # load_Data2Txt3cls(origin_path, './')

    # size = (32, 64, 64)
    # t = transforms.Compose([
    #     transforms.MRINormalize(),
    #     transforms.Resize3D(target_size=size)
    # ])
    # control_txt_path = "./Control_data_list.txt"
    # control_dataset = PPMI_dataset(control_txt_path, transform=t)
    # pd_txt_path = "./PD_data_list.txt"
    # pd_dataset = PPMI_dataset(pd_txt_path, transform=t)
    #
    # kf = KFold(n_splits=5, shuffle=True, random_state=1)
    # fold = 0
    # for (c_train_index, c_val_index), (p_train_index, p_val_index) in zip(kf.split(control_dataset),
    #                                                                       kf.split(pd_dataset)):
    #     fold += 1
    #     print(f"Fold {fold}")
    #     c_train_subset = Subset(control_dataset, c_train_index)
    #     c_val_subset = Subset(control_dataset, c_val_index)
    #     p_train_subset = Subset(pd_dataset, p_train_index)
    #     p_val_subset = Subset(pd_dataset, p_val_index)
    #
    #     train_subet = data_utils.ConcatDataset([c_train_subset, p_train_subset])
    #     val_subet = data_utils.ConcatDataset([c_val_subset, p_val_subset])
    #     print(len(c_train_subset), len(p_train_subset), len(train_subet))
    #     print(len(c_val_subset), len(p_val_subset), len(val_subet))
    # train_loader = DataLoader(train_subset, batch_size=, shuffle=True)
    # val_loader = DataLoader(val_subset, batch_size=, shuffle=True)
    # load_Data2Txt(origin_path)

    img_path_list, labels_list = loadTxt2List(txt_path)
    # print(img_path_list)
    import monai


    resize_transform = mtf.Resize((32, 256, 256))
    intensity_transform = mtf.ScaleIntensity()

    train_transform = mtf.Compose(
        [
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            resize_transform,
            intensity_transform,
            mtf.ToTensor(dtype=torch.float),
        ]
    )
    import tqdm
    from torch.utils.data import DataLoader

    # data_set = PPMI_dataset_with_Monai_SMOTE(img_path_list, labels_list, transforms=train_transform)
    # data_set=PPMI_T1dataset_with_dataAug(img_path_list, labels_list, is_training=True)
    data_set=PPMI_dataset_with_Monai(img_path_list, labels_list, is_training=True,transforms=train_transform)
    train_loader = DataLoader(data_set, batch_size=1, shuffle=True)
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader, desc='Training', dynamic_ncols=False)):
        # print(batch_idx)
        print(data.shape)
        pass
    print(data_set[0][0].shape)
