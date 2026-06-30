"""
file: M3D_TransMIL_
author:
create time: 2024/9/18
last modified: 2024/9/18/12:32
version: 1.0
description: prompt的baseline的M3DCLIP
"""

import transformers
import torch.nn.functional as F

import torch
import torch.nn as nn
import os
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
import models.MedicalNet as cnnmodel
from nystrom_attention import NystromAttention
class preModel(nn.Module):
    def __init__(self, device):
        super(preModel, self).__init__()
        self.device = device
        net = cnnmodel.resnet10(
            sample_input_W=80,
            sample_input_H=96,
            sample_input_D=80,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=2).to(device)
        net = torch.nn.DataParallel(net)
        checkpoint_path = r'D:\fftprogramme\Python\Model\MedicalNet\resnet_10.pth'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        self.net = net.module
    def forward(self, x):
        out=self.net(x)
        out=out.flatten(2).transpose(-1, -2)
        return out

class Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()

    def forward(self, x, w, bias, gamma):
        out = x.contiguous().view(x.size(0) * x.size(1), x.size(2))

        out_f = F.linear(out, w, bias)

        out = out_f.view(x.size(0), x.size(1), out_f.size(1))

        # 范数应用
        out = torch.sqrt((out ** 2).sum(2))
        # 归一化
        alpha_01 = out / out.sum(1, keepdim=True).expand_as(out)

        alpha_01 = F.relu(alpha_01 - 0.1 / float(gamma))

        # 再次将alpha归一化
        alpha_01 = alpha_01 / alpha_01.sum(1, keepdim=True).expand_as(alpha_01)

        alpha = torch.unsqueeze(alpha_01, dim=2)
        out = alpha.expand_as(x) * x

        return out, out_f, alpha_01

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512,dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        # x=self.norm(x)
        out=self.attn(x)
        x = x + out
        return x


class EnhancedAdapter(nn.Module):
    def __init__(self, dim, reduction=4):  # 增大降维系数
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.norm = nn.LayerNorm(dim // reduction)  # 前置归一化
        self.up = nn.Linear(dim // reduction, dim)
        self.gate = nn.Parameter(torch.ones(1))  # 动态门控

    def forward(self, x):
        identity = x
        x = self.down(x)
        x = self.norm(x)  # 前置归一化稳定中间特征
        x = F.gelu(x)  # 替换激活函数
        return identity + self.gate * self.up(x)  # 可学习门控

class Adapter(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)
        self.layer_norm = nn.LayerNorm(dim // reduction)
    def forward(self, x):
        identity = x
        x = self.down(x)
        x = self.layer_norm(x)  # 前置归一化稳定中间特征
        x = F.gelu(x)  # 替换激活函数
        return identity + self.up(x)  # 可学习门控


class Transformers(nn.Module):
    def __init__(self, device=torch.device("cuda"), modelpath=r"D:\fftprogramme\Python\Model\LLM\M3D-CLIP",
                 embed_dropout=0):
        super(Transformers, self).__init__()
        self.device = device
        path = modelpath
        self.M3D_CLIP = transformers.AutoModel.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            trust_remote_code=True,
            cache_dir=path,
            # dtype=dtype
        ).to(device).requires_grad_(False)
        self.in_dim = 512
        self.out_dim = 768
        self.preModel = preModel(device)
        # self.trans_layer=TransLayer(dim=768)
        # lossAttention
        self.attention_layer = Attention_Layer()
        self._fc1 = nn.Linear(self.out_dim, 2)
        self.N_patches = 10 * 12 * 10
        self.pos_embed = build_sincos_position_embedding([10, 12, 10], self.out_dim, 3)
        # self.pos_embed = nn.Parameter(torch.randn(1, self.N_patches+1, self.out_dim)).to(device)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.out_dim)).to(device)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.blocks = self.M3D_CLIP.vision_encoder.blocks
        self._fc2 = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.out_dim)).to(device)
        self.n_classes = 2
        self.adapter = Adapter(self.out_dim,reduction=4)
        self.layerNorm = nn.LayerNorm(self.out_dim)
        self.classifier = nn.Linear(self.out_dim, self.n_classes)

    def forward(self, x):
        # [B,N,dim=384]
        CNN_features = self.preModel(x)
        # lossAttention

        # gamma = CNN_features.size(1)
        # out_lb, f, alpha = self.attention_layer(CNN_features, self._fc1.weight, self._fc1.bias, gamma)
        # out_lb = out_lb.transpose(2, 1).sum(2)
        # out_lb = out_lb.view(out_lb.size(0), -1)
        # out_lb = self._fc1(out_lb)
        out_ft = self._fc2(CNN_features)
        # out_ft = CNN_features
        B = out_ft.shape[0]
        # pos_embed = self.pos_embed
        pos_embed = torch.cat((self.cls_pos_embed, self.pos_embed), dim=1)
        out_ft = torch.cat((self.cls_token.expand(B, -1, -1), out_ft), dim=1)
        out_ts = out_ft + pos_embed
        # out_ts = self.embed_dropout(out_ts)
        # 添加位置信息
        for block in self.blocks:
            out_ts = block(out_ts)
            out_ts=self.adapter(out_ts)
        # out_ts = self.trans_layer(out_ts)
        cls_token = out_ts[:, 0]
        # cls_token = self.adapter(cls_token)
        cls_token = self.layerNorm(cls_token)
        logits = self.classifier(cls_token)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_hat, Y_prob


if __name__ == "__main__":
    # from loss.weight_loss import CrossEntropyLoss as CE

    device = torch.device("cuda:0")
    data = torch.randn((4, 1, 80, 96, 80))
    label = torch.tensor([0, 1, 0, 1]).to(device)
    data = data.cuda()
    # preProcessModel=preProcessModel().to(device)
    # output=preProcessModel(data)
    # print(output.shape)
    model = Transformers(device=device).to(device)
    model = model.cuda()
    # print(model.eval())
    logits, Y_hat, Y_prob= model(data)
    criterion1 = torch.nn.CrossEntropyLoss(
        reduction='none', size_average=True).to(device)
    # weight_criterion = CE(aggregate='sum').to(device)
    # loss_2 = criterion1(out_lb, label)
    # loss_3 = weight_criterion(out_f, label.repeat(alpha.size(1), 1).permute(1, 0).contiguous().view(-1),
    #                           weights=alpha.view(-1))
    print(logits, Y_hat, Y_prob)
    # print(results_dict)
    # Modeleval()
    # from torchsummary import summary

    # summary(model, input_size=((1, 160, 192, 160)))
    # Modeleval()
