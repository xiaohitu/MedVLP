"""
file: train_baseline_Attention3DPPEG.py
author: 
create time: 2025/2/20
last modified: 2025/2/20/14:29
version: 1.0
description: 
"""

from __future__ import print_function

import random
import monai
from utils.Utils import saveXslx_train, SaveBestModel, saveBestInfo, showPlt_train
from utils import Utils as staticsUtils
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from PPMI import PPMI_3D_dataset as dataset_utils
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
import os
import argparse
import monai.transforms as mtf
from pytorch_toolbelt import losses as L
import torch.nn.functional as F
from loss.weight_loss import CrossEntropyLoss as CE
from models.LA_LB_M3D_Base import Transformers


def train(args, data_list, label_list, transforms):
    # from model_with_clip_feature import CNNwithCLIP
    checkpoint_dir = r'./checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.epochs = 80
    args.lr = 0.0001
    args.batch_size_train = 8
    args.batch_size_val = 8
    args.patience = args.epochs
    # args.model = f"(PDT2AugBLData(Control-PD))M3D_TM_AttPPEG[xhcDrop=(0.1)B={args.batch_size_train}dim=512]"
    args.model = f"(ADNI(NC-PD))LALB_M3D_Base[B={args.batch_size_train}]"
    print(f"Project:{args.model}")
    device = torch.device("cuda:" + str(args.device))
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    num_0 = np.sum(label_list == 0)
    num_1 = np.sum(label_list == 1)
    loss_weight = torch.Tensor([num_0, num_1])
    loss_weight /= loss_weight.sum()
    # 五折训练
    skf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_global_checkpoint.pth")
    # 检查是否存在全局检查点
    if os.path.exists(checkpoint_path):
        print(f"Found global checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_fold = checkpoint['fold']
        print(f"Resuming from fold {start_fold}")
    else:
        start_fold = 0  # 从第0个fold开始
    recent_checkpoint_file = []
    for fold, (train_index, val_index) in enumerate(skf.split(data_list, label_list), start=1):
        # 如果当前 fold 已完成，跳过
        if fold < start_fold:
            print(f"Skipping fold {fold} (already completed)")
            continue
        print(f"Fold {fold}---------------starting-----------------------")
        # 保存每一折的最好模型
        save_best_model_callback = SaveBestModel(target=0.5, path=save_dir + '/' + str(
            args.model + '_' + str(fold) + 'fold_best_model'), verbose=1)
        # 初始化早停机制
        early_stopping = staticsUtils.EarlyStopping(patience=args.patience, delta=0.001, mode='max', save_best=True)
        train_subset = dataset_utils.PPMI_T1dataset_withAugMonai(data_list[train_index], label_list[train_index],
                                                                 is_training=True, transforms=transforms["train"])
        val_subset = dataset_utils.PPMI_T1dataset_withAugMonai(data_list[val_index], label_list[val_index],
                                                               is_training=False, transforms=transforms["val"])
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size_train, shuffle=True)
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size_val, shuffle=True)

        print('Init Model, using model:' + args.model)
        model = Transformers(device=device)
        criterion_xhc = torch.nn.CrossEntropyLoss(
            weight=loss_weight, reduction='none', size_average=True)
        criterion_focalLoss = L.BinaryFocalLoss(alpha=0.76)
        criterion1 = torch.nn.CrossEntropyLoss(
            reduction='none', size_average=True)
        weight_criterion = CE(aggregate='sum')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(
            0.9, 0.999), weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
        # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.reg)
        # 可以加上scheduler进行训练
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250], gamma=0.5)
        if args.cuda:
            model = model.to(device)
            criterion1 = criterion1.to(device)
            criterion_xhc = criterion_xhc.to(device)
            criterion_focalLoss = criterion_focalLoss.to(device)
            weight_criterion = weight_criterion.to(device)

        train_args_list = []
        validation_args_list = []
        val_yhat_list = []
        val_labels_list = []
        val_prob_list = []
        # 如果从该 fold 开始，则加载全局检查点的状态
        if fold == start_fold and os.path.exists(checkpoint_path):
            # 只恢复当前 fold 的 epoch
            start_epoch = checkpoint['epoch']
            save_best_model_callback.set_state(checkpoint['save_best_model_state'])
            early_stopping.set_state(checkpoint['early_stopping_state'])
            print(f"Loading model and optimizer state from checkpoint for fold {fold} epoch {start_epoch}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch += 1
            # 恢复存储的列表数据
            train_args_list = checkpoint.get('train_args_list', [])
            validation_args_list = checkpoint.get('validation_args_list', [])
            val_yhat_list = checkpoint.get('val_yhat_list', [])
            val_labels_list = checkpoint.get('val_labels_list', [])
            val_prob_list = checkpoint.get('val_prob_list', [])
        else:
            start_epoch = 0

        # 训练模型
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss = 0.
            train_correct_predict = 0
            train_total_samples = 0
            for batch_idx, (data, label) in enumerate(tqdm(train_loader, desc='Training', dynamic_ncols=True,
                                                           leave=True)):

                if args.cuda:
                    data = data.to(device)
                    label = label.to(device)
                data, label = Variable(data), Variable(label)
                # print(data.shape)
                label = label.type(torch.LongTensor).to(device)

                # print(y.device(),label.device)
                # instance_labels = label*torch.squeeze(torch.ones(data.size(0),1)).type(torch.LongTensor)
                # reset gradients
                optimizer.zero_grad()
                # calculate loss and metrics
                logits, Y_hat, Y_prob, out_lb, out_f, alpha = model(data)

                loss_1 = criterion1(logits, label)
                loss_2 = criterion1(out_lb, label)
                loss_3 = weight_criterion(out_f, label.repeat(alpha.size(1), 1).permute(1, 0).contiguous().view(-1),
                                          weights=alpha.view(-1))
                loss = loss_1 + loss_2 + loss_3
                train_loss += loss.item()

                predicted_labels = Y_hat  # 获取最大概率的索引作为预测标签
                # 计算预测正确的样本数量
                train_correct_predict += (predicted_labels ==
                                          label).sum().item()
                train_total_samples += label.size(0)
                # backward pass
                loss.backward()
                # step
                optimizer.step()

            # calculate loss and error for epoch
            train_loss /= len(train_loader)
            train_acc = train_correct_predict / train_total_samples

            # 验证模型
            model.eval()
            val_predictions = []
            val_labels = []
            # 初始化一个空列表来收集预测概率
            val_predictions_prob = []

            val_loss = 0.
            val_correct_predict = 0
            val_total_samples = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(tqdm(val_loader, desc='Testing', dynamic_ncols=True,
                                                               leave=True)):
                    # instance_labels=label
                    if args.cuda:
                        data = data.to(device)
                        label = label.to(device)
                    data, label = Variable(data), Variable(label)

                    label = label.type(torch.LongTensor).to(device)

                    logits, Y_hat, Y_prob, _, _, _ = model(data)
                    loss = criterion1(logits, label)

                    val_loss += loss.item()

                    # 将预测概率移动到 CPU 并转换为 NumPy 数组
                    val_predictions_tensor = Y_prob[:, 1]
                    # 将y_pred转换为numpy数组并添加到列表
                    val_predictions_prob.extend(val_predictions_tensor.cpu().numpy())

                    # 获取最大概率的索引作为预测标签
                    predicted_labels = Y_hat
                    # 将y_true转换为numpy数组并添加到列表
                    val_labels.extend(label.cpu().numpy())
                    val_predictions.extend(predicted_labels.cpu().numpy())
                    val_correct_predict += (predicted_labels == label).sum().item()
                    # 更新总样本数和正确预测的样本数
                    val_total_samples += label.size(0)
            val_loss /= len(val_loader)
            val_acc = val_correct_predict / val_total_samples
            scheduler.step()
            # 更新早停状态
            stop_training = early_stopping.step(val_acc, epoch)
            # 保存当前epoch的状态
            logs = {
                'fold': fold,
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_yhat_list': val_predictions,
                'val_labels_list': val_labels,
                'val_prob_list': val_predictions_prob,
            }
            train_args_list.append((train_loss, train_acc))
            validation_args_list.append((val_loss, val_acc))
            val_yhat_list.append(val_predictions)
            val_labels_list.append(val_labels)
            val_prob_list.append(val_predictions_prob)

            save_best_model_callback(epoch, model, logs)

            tqdm.write(
                'Fold:{},Epoch: {}/{}, Train Loss: {:.4f}, Train Acc: {:.4f}, val Loss: {:.4f},  val Acc: {'
                ':.4f}'.format(
                    str(fold), epoch +
                               1, args.epochs, train_loss, train_acc, val_loss, val_acc
                ))
            saveBestInfo(logs, log_name=args.model + "train_log")

            # 保存全局检查点
            checkpoint = {
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_args_list': train_args_list,
                'validation_args_list': validation_args_list,
                'val_yhat_list': val_yhat_list,
                'val_labels_list': val_labels_list,
                'val_prob_list': val_prob_list,
                'early_stopping_state': early_stopping.get_state(),
                'save_best_model_state': save_best_model_callback.get_state()
            }
            staticsUtils.save_checkpoint_atomically(checkpoint, checkpoint_path)
            # 停止训练
            if stop_training:
                print(f"Early stopping triggered at fold {fold}, epoch {epoch}")
                break

        # 生成plt图表
        showPlt_train(str(args.model) + str(fold) +
                      "th Cross_validation", train_args_list, validation_args_list)
        # 保存训练记录
        # print(len(val_predictions_list),len(val_labels_list))
        saveXslx_train(str(args.model) + str(fold) + "th Cross_validation", train_args_list, validation_args_list,
                       val_yhat_list, val_labels_list, val_prob_list)


def make_parse():
    parser = argparse.ArgumentParser(description='M3D_CLIP TranMIL PPMI')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='S',
                        help='batch_size')
    parser.add_argument('--device', type=int, default=0, metavar='D',
                        help='gpu (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='using CUDA training')
    parser.add_argument('--model', type=str, default='attention',
                        help='Choose b/w attention and gated_attention')
    args = parser.parse_args()
    return args


def nload_Data2Txt(origin_path=r"D:\data\PPMI\T1_align", txt_path='./', classifier=["Prodromal", "PD"]):
    # 保证可复现
    # random.seed(1)
    data_path = []
    for label, cls in enumerate(classifier):
        subject_path = os.path.join(origin_path, cls)
        for file in os.listdir(subject_path):
            path = os.path.join(subject_path, file)
            data_path.append((path, label))

    # 使用 with open 写入 train_list.txt
    name = "PDT1[pro-PD]data_list.txt"
    with open(os.path.join(txt_path, name), 'w') as train_txt:
        for item in data_path:
            # 假设元组中的元素都是字符串，可以直接连接
            line = item[0] + ' ' + str(item[1]) + '\n'
            train_txt.write(line)

    print("dataset load over! The files are saved in " + txt_path + name)


if __name__ == "__main__":

    args = make_parse()

    # 定义最佳函数回调
    save_dir = './save_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # origin_path = r"D:\data\FreesufferAlignOutdataset"
    # origin_path = r"D:\data\PPMI\T1_fslAlignBet"
    origin_path = r"F:\Alzheimer_Dataset\ADNI1"
    txt_path = "./ADdata_list.txt"
    # txt_path = "./PDT2[Pro-PDSelected]data_list.txt"
    # nload_Data2Txt(origin_path)
    # dataset_utils.load_Data2Txt(origin_path)
    img_path_list, labels_list = dataset_utils.loadTxt2List(txt_path)
    # for img,label in zip(img_path_list,labels_list):
    #     if label==0:
    #         img_path_list.append(img)
    #         labels_list.append(0)
    # mtf.utils.set_determinism(seed=12345)
    seed = 42
    args.seed = seed
    # 设置随机数
    # 设置 Python 随机数生成器种子
    random.seed(seed)

    # 设置 NumPy 随机数生成器种子
    np.random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    monai.utils.set_determinism(seed)

    # 定义数据增强
    resize_transform = mtf.Resize((32, 256, 256))
    intensity_transform = mtf.ScaleIntensity(minv=0.0, maxv=1.0)

    train_transform = mtf.Compose(
        [

            # 加性噪声
            # mtf.RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
            # mtf.RandAdjustContrast(prob=0.2, gamma=(0.5, 1.5)),
            # 调整尺寸
            resize_transform,
            # 强度缩放
            intensity_transform,
            # 转换为张量
            mtf.ToTensor(dtype=torch.float),
        ]
    )

    val_transform = mtf.Compose(

        [
            resize_transform,
            intensity_transform,
            mtf.ToTensor(dtype=torch.float),
        ]
    )
    transforms = {"train": train_transform, "val": val_transform}
    train(args, data_list=img_path_list,
          label_list=labels_list, transforms=transforms)
