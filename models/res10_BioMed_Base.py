"""
file:
author:
create time: 2024/9/18
last modified: 2024/9/18/12:32
version: 1.0
description:
"""
import json
import os

# import transformers
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import torch
import torch.nn as nn
import MedicalNet as cnnmodel
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding


class preModel(nn.Module):
    def __init__(self, device):
        super(preModel, self).__init__()
        self.device = device
        net = cnnmodel.resnet10(
            sample_input_W=160,
            sample_input_H=192,
            sample_input_D=160,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=2).to(device)
        net = torch.nn.DataParallel(net)
        checkpoint_path = r'D:\fftprogramme\Python\Model\MedicalNet\resnet_10.pth'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        self.net = net.module

        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    def forward(self, x):
        out = self.net(x)
        out=self.pool(out)
        out = out.flatten(2).transpose(-1, -2)
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


class Transformers(nn.Module):
    def __init__(self, device=torch.device("cuda"),
                 modelpath=r"D:\fftprogramme\Python\Model\biomedclip\BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super(Transformers, self).__init__()
        self.device = device
        # 加载模型
        model_name = "biomedclip_local"
        with open(os.path.join(modelpath, "open_clip_config.json"), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            # preprocess_cfg = config["preprocess_cfg"]

        if (not model_name.startswith(HF_HUB_PREFIX)
                and model_name not in _MODEL_CONFIGS
                and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        self.bio_CLIP, _, preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=os.path.join(modelpath, "open_clip_pytorch_model.bin"))
        self.bio_CLIP = self.bio_CLIP.to(device).requires_grad_(False)
        # 获得模型的blocks
        self.blocks = self.bio_CLIP.visual.trunk.blocks.requires_grad_(False)
        # tokenizer = get_tokenizer(model_name)
        self.in_dim = 512
        self.out_dim = 768
        self.preModel = preModel(device)

        # lossAttention
        # self.attention_layer = Attention_Layer()
        # self._fc1 = nn.Linear(256, 2)
        self.N_patches = 10 * 12 * 10
        self.pos_embed = build_sincos_position_embedding([10, 12, 10], self.out_dim, 3)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.out_dim)).to(device)
        self._fc2 = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.out_dim)).to(device)
        self.n_classes = 2

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
        # CNN_features = self.pool(CNN_features)
        out_ft = self._fc2(CNN_features)

        B = out_ft.shape[0]
        pos_embed = torch.cat((self.cls_pos_embed, self.pos_embed), dim=1)
        out_ft = torch.cat((self.cls_token.expand(B, -1, -1), out_ft), dim=1)
        out_ts = out_ft + pos_embed
        # 添加位置信息
        for block in self.blocks:
            out_ts = block(out_ts)
        cls_token = out_ts[:, 0]
        cls_token = self.layerNorm(cls_token)
        logits = self.classifier(cls_token)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_hat, Y_prob


if __name__ == "__main__":
    # from loss.weight_loss import CrossEntropyLoss as CE

    device = torch.device("cuda:0")
    data = torch.randn((4, 1, 160, 192, 160))
    label = torch.tensor([0, 1, 0, 1]).to(device)
    data = data.cuda()
    # preProcessModel=preProcessModel().to(device)
    # output=preProcessModel(data)
    # print(output.shape)
    model = Transformers(device=device).to(device)
    model = model.cuda()
    # print(model.eval())
    # logits, Y_hat, Y_prob, out_lb, out_f, alpha = model(data)
    logits, Y_hat, Y_prob = model(data)
    # criterion1 = torch.nn.CrossEntropyLoss(
    #     reduction='none', size_average=True).to(device)
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
