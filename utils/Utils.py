"""
file: Utils
author: 
create time: 2024/11/25
last modified: 2024/11/25/16:09
version: 1.0
description: 
"""
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def saveXslx_train(title, train_args_list, validation_args_list, val_yhat_list, val_labels_list, val_prob_list):
    # 提取训练集的损失和准确率
    train_loss = [item[0] for item in train_args_list]
    train_acc = [item[1] for item in train_args_list]

    # 提取验证集的损失和准确率
    val_loss = [item[0] for item in validation_args_list]
    val_acc = [item[1] for item in validation_args_list]

    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    f1 = []
    auc = []
    for y_true, y_hat, y_prob in zip(val_labels_list, val_yhat_list, val_prob_list):
        conf_matrix = confusion_matrix(y_true, y_hat)
        accuracy.append(accuracy_score(y_true, y_hat))
        precision.append(precision_score(y_true, y_hat, average='binary', zero_division=1))
        sensitivity.append(recall_score(y_true, y_hat, average='binary'))
        TN = conf_matrix[0, 0]  # 假设类别 0 是负类
        FP = conf_matrix[0, 1]  # FP 是所有非对角线元素的和
        specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)  # 防止除以零
        # 计算 F1-score
        f1.append(f1_score(y_true, y_hat, average='binary'))
        auc.append(roc_auc_score(y_true, y_prob))
    # 创建DataFrame
    data = {
        "Epoch": range(1, len(train_loss) + 1),
        "auc": auc,
        "Accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
    }
    df = pd.DataFrame(data)
    # 保存为xlsx文件
    file_path = f'./output/{title.replace(" ", "_")}_results.xlsx'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Train Results', index=False)
    print(f'Excel saved as: {file_path}')
    return file_path


def saveXslx_train_3cls(title, train_args_list, validation_args_list, val_yhat_list, val_labels_list, val_prob_list):
    # 提取训练集的损失和准确率
    train_loss = [item[0] for item in train_args_list]
    train_acc = [item[1] for item in train_args_list]

    # 提取验证集的损失和准确率
    val_loss = [item[0] for item in validation_args_list]
    val_acc = [item[1] for item in validation_args_list]

    accuracy = []
    precision = []
    sensitivity = []  # 这里等同于 recall
    specificity = []
    f1 = []
    auc = []

    for y_true, y_hat, y_prob in zip(val_labels_list, val_yhat_list, val_prob_list):
        # 混淆矩阵
        conf_matrix = confusion_matrix(y_true, y_hat)

        # 计算常规指标
        accuracy.append(accuracy_score(y_true, y_hat))
        precision.append(precision_score(y_true, y_hat, average='macro', zero_division=1))
        sensitivity.append(recall_score(y_true, y_hat, average='macro'))
        f1.append(f1_score(y_true, y_hat, average='macro'))

        # 计算 specificity: 针对每个类的特异性
        class_specificity = []
        for i in range(conf_matrix.shape[0]):
            TN = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificity_i = TN / (TN + FP) if (TN + FP) > 0 else 0
            class_specificity.append(specificity_i)
        specificity.append(np.mean(class_specificity))

        # 计算 AUC: 使用 one-vs-rest 方法
        y_true_one_hot = np.eye(len(np.unique(y_true)))[y_true]  # 转为 One-Hot 编码
        auc.append(roc_auc_score(y_true_one_hot, y_prob, average='macro', multi_class='ovr'))

    # 创建 DataFrame
    data = {
        "Epoch": range(1, len(train_loss) + 1),
        "auc": auc,
        "Accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
    }
    df = pd.DataFrame(data)

    # 保存为 xlsx 文件
    # todo 保存文件的时候应该改为Try有异常的，增加健壮性
    file_path = f'./output/{title.replace(" ", "_")}_results.xlsx'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Train Results', index=False)

    print(f'Excel saved as: {file_path}')
    return file_path

def saveXslx_test(title, test_yhat_list, test_labels_list, test_prob_list):
    # 提取验证集的损失和准确率
    # test_loss = [item[0] for item in test_args_list]
    # test_acc = [item[1] for item in test_args_list]

    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    f1 = []
    auc = []
    for y_true, y_hat, y_prob in zip(test_labels_list, test_yhat_list, test_prob_list):
        conf_matrix = confusion_matrix(y_true, y_hat)
        accuracy.append(accuracy_score(y_true, y_hat))
        precision.append(precision_score(y_true, y_hat, average='binary', zero_division=1))
        sensitivity.append(recall_score(y_true, y_hat, average='binary'))
        TN = conf_matrix[0, 0]  # 假设类别 0 是负类
        FP = conf_matrix[0, 1]  # FP 是所有非对角线元素的和
        specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)  # 防止除以零
        # 计算 F1-score
        f1.append(f1_score(y_true, y_hat, average='binary'))
        auc.append(roc_auc_score(y_true, y_prob))
    # 创建DataFrame
    data = {
        "epoch": range(1, len(accuracy) + 1),
        "auc": auc,
        "Accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }
    df = pd.DataFrame(data)
    # 保存为xlsx文件
    file_path = f'./output/{title.replace(" ", "_")}_results.xlsx'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Train Results', index=False)
    print(f'Excel saved as: {file_path}')
    return file_path

def saveBestInfo(logs: dict, log_name):
    path = "./logs"
    os.makedirs(path, exist_ok=True)
    name = os.path.join(path, log_name + ".txt")

    with open(name, 'a') as write:
        # 遍历字典中的每个项，并写入文件
        for key, value in logs.items():
            # 将字典中的键和值转换为字符串，并追加一个换行符
            if key == "val_yhat_list" or key == "val_labels_list" or key == "val_prob_list" or key == "save_best_model_state":
                continue
            write.write(f"{key}: {value} ")
        write.write("\n")


class SaveBestModel:
    def __init__(self, target=0.5, path='../best_model', verbose=0, exclude_prefixes=["M3D_CLIP", "bio_CLIP","blocks",
                 "preModel","ds_model"]):
        """
        Args:
            target (float): 初始目标值（如验证集的最高准确率）。
            path (str): 保存的文件路径。
            verbose (int): 是否打印详细日志信息，1 为打印，0 为不打印。
            exclude_prefix (str): 需要排除的模块前缀（如 "M3DCLIP"）。
        """
        self.target = target
        self.epoch = None
        self.path = path
        self.verbose = verbose
        self.exclude_prefixes = exclude_prefixes

    def __call__(self, epoch, model, logs):
        """
        保存最佳模型权重（排除特定模块）。

        Args:
            epoch (int): 当前训练的 epoch。
            model (torch.nn.Module): 当前模型。
            logs (dict): 当前 epoch 的日志信息（包括 'val_acc' 等）。
        """
        if logs['val_acc'] > self.target:
            self.target = logs['val_acc']
            self.epoch = epoch

            # 过滤掉指定模块的权重
            state_dict = model.state_dict()
            if self.exclude_prefixes:
                # 使用 any() 检查多个前缀
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not any(prefix in k for prefix in self.exclude_prefixes)
                }
            else:
                filtered_state_dict = state_dict
            state = {}
            for k, v in logs.items():
                state[k] = v
            # 保存模型权重和日志
            state['state_dict'] = filtered_state_dict
            torch.save(state, f"{self.path}_epoch{epoch}.pt")

            if self.verbose:
                print(
                    f"Best val_acc updated to {self.target:.4f} at epoch {self.epoch}. Model saved without '"
                    f"{self.exclude_prefixes}' weights.")
                log_name = os.path.basename(self.path)
                saveBestInfo(logs, log_name)

    def get_state(self):
        """获取当前状态"""
        return {
            'target': self.target,
            'epoch': self.epoch,
            'path': self.path,
            'verbose': self.verbose
        }

    def set_state(self, state):
        """设置状态"""
        self.target = state['target']
        self.epoch = state['epoch']
        self.path = state['path']
        self.verbose = state['verbose']


def showPlt_train(title, train_args_list, validation_args_list):
    train_loss = [item[0] for item in train_args_list]
    train_acc = [item[1] for item in train_args_list]

    # 提取验证集的损失、准确率
    val_loss = [item[0] for item in validation_args_list]
    val_acc = [item[1] for item in validation_args_list]
    # val_error = [item[2] for item in validation_args_list]
    # 创建一个画布，设置大小
    plt.figure(figsize=(15, 5))

    # 绘制损失图
    plt.subplot(1, 2, 1)  # 1行2列的第1个
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率图
    plt.subplot(1, 2, 2)  # 1行2列的第2个
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    # 计算平均准确率和最高准确率
    avg_val_acc = sum(val_acc) / len(val_acc)
    max_val_acc = max(val_acc)

    # 在图表上显示平均准确率和最高准确率
    plt.text(0, 1.05, f'Avg Val Acc: {avg_val_acc:.4f}', transform=plt.gca().transAxes, fontsize=12, color='red')
    plt.text(0, 1.15, f'Max Val Acc: {max_val_acc:.4f}', transform=plt.gca().transAxes, fontsize=12, color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 给整张图表设置标题
    plt.suptitle(title, fontsize=16, y=0.95)  # y=1.02 是为了让标题位于图表的顶部

    # 显示图表
    plt.tight_layout()

    save_path = './output/plot'
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存图表为 PNG 文件
    save_filename = f'{title.replace(" ", "_")}_plot.png'
    save_path = os.path.join(save_path, save_filename)
    plt.savefig(save_path, dpi=100)  # 可以根据需要调整 DPI
    # plt.show()
    # # 清除图形状态，以便下次使用
    # plt.clf()

    print(f'Plot saved as: {save_path}')


class EarlyStopping:
    def __init__(self, patience=10, delta=0, mode='max', save_best=True):
        """
        早停机制类

        Args:
            patience (int): 多少个epoch没有改善则触发早停，默认10
            delta (float): 改善的最小阈值，默认0
            mode (str): 监控指标是最大化还是最小化，默认'max'
            save_best (bool): 是否保存最佳模型，默认True
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_best = save_best

        # 初始化状态
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.wait = 0
        self.stop_training = False

    def step(self, current_value, epoch):
        """
        更新早停状态

        Args:
            current_value (float): 当前epoch的监控指标值
            epoch (int): 当前epoch

        Returns:
            bool: 是否触发早停
        """
        if self.mode == 'max':
            improved = current_value > self.best_value + self.delta
        else:
            improved = current_value < self.best_value - self.delta

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stop_training = True

        return self.stop_training

    def get_state(self):
        """获取当前状态"""
        return {
            'best_value': self.best_value,
            'wait': self.wait,
            'stop_training': self.stop_training
        }

    def set_state(self, state):
        """设置状态"""
        self.best_value = state['best_value']
        self.wait = state['wait']
        self.stop_training = state['stop_training']


def save_checkpoint_atomically(checkpoint, checkpoint_path, exclude_prefixes=["M3D_CLIP","bio_CLIP", "blocks",
                                                                              "preModel","ds_model"]):
    temp_path = checkpoint_path + '.tmp'
    # 过滤掉指定模块的权重
    state_dict = checkpoint['model_state_dict']
    if exclude_prefixes:
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(prefix in k for prefix in exclude_prefixes)
        }
    else:
        filtered_state_dict = state_dict
    checkpoint['model_state_dict'] = filtered_state_dict
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path} without '{exclude_prefixes}' weights.")
