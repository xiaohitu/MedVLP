"""
file: res10_M3D_patchPrompt
author: 
create time: 2025/3/28
last modified: 2025/3/28/1:28
version: 1.0
description: 
"""

import transformers
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import models.MedicalNet as cnnmodel
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding


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
        # net = torch.nn.DataParallel(net)
        checkpoint_path = r'D:\fftprogramme\Python\Model\MedicalNet\resnet_10.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        self.net = net

    def forward(self, x):
        out = self.net(x)
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


# patch_text_description = [
#     "Normal brain volume.",
#     "Uniform cortical thickness.",
#     "Symmetrical brain structure.",
#     "Normal ventricular morphology.",

#     "Significant global brain volume reduction.",
#     "Significant ventricular enlargement.",
#     "Pronounced hippocampal atrophy.",
#     "Marked local cortical thinning."
# ]

patch_text_description = [
    "Normal,age-appropriate iron levels",
    "Normal,no focal iron accumulation",
    "Normal,preserved neuromelanin signal on NM-MRI",
    "Normal,intact 'swallow tail sign'",

    # "Focal iron accumulation",
    # "Patchy hypointensity in anteromedial globus pallidus",
    # "Subtle dorsolateral volume loss",
    # "Partial loss of dorsolateral hyperintensity",

    "Significant,diffuse iron overload",
    "Significant,severe focal iron deposition",
    "Significant,near-complete loss of neuromelanin signal",
    "Significant,complete 'swallow tail sign' absence",
]


class PPMIPromptEncoder(nn.Module):
    def __init__(self, model, tokenizers, device="cuda:0"):
        super().__init__()
        # 注册CLIP模型组件
        self.tokenizer = tokenizers
        self.device = device
        self.model = model
        # 可配置参数

        text_inputs = patch_text_description

        text_tensor = self.tokenizer(
            text_inputs,
            return_tensors="pt",  # 返回PyTorch张量格式（可选，根据框架需求调整）
            padding=True,  # 自动填充到最大序列长度（若需批量处理）
            truncation=True  # 自动截断到模型最大长度（防止超限）
        )
        self.input_id = text_tensor["input_ids"].to(device=self.device)
        self.attention_mask = text_tensor["attention_mask"].to(device=self.device)

        self.init_embed = model.encode_text(
            self.input_id, self.attention_mask)

    def forward(self):
        """生成可学习提示的嵌入表示"""
        patch_features = self.init_embed[:, 1:-1, :].mean(dim=1)
        # patch_features = self.proj(patch_features)
        return patch_features


class Adapter(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


class Transformers(nn.Module):
    def __init__(self, device=torch.device("cuda"), modelpath1=r"D:\fftprogramme\Python\Model\LLM\M3D-CLIP"):
        super(Transformers, self).__init__()
        self.device = device
        m3d_path = modelpath1
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            cache_dir=m3d_path,
            # dtype=dtype
        )

        self.M3D_CLIP = transformers.AutoModel.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            trust_remote_code=True,
            cache_dir=m3d_path,
            # dtype=dtype
        ).to(device).requires_grad_(False)
        self.n_classes = 2
        self.in_dim = 512
        self.out_dim = 768
        # PatchPreProcess
        # self.CNN = LA_GMF(device)
        self.preModel = preModel(device)
        # 初始阶段空间对齐
        self.i_izen = nn.Parameter(torch.randn(1, 512, self.out_dim))
        self.t_izen = nn.Parameter(torch.randn(1, self.out_dim, self.out_dim))
        # 最终阶段空间对齐
        self.W_Img = nn.Parameter(torch.randn(1, self.out_dim, self.out_dim))
        self.W_Text = nn.Parameter(torch.randn(1, self.out_dim, self.out_dim))
        # lossAttention
        # self.attention_layer = Attention_Layer()
        # self._fc1 = nn.Linear(256, 2)

        # 大模型
        self.N_patches = 10 * 12 * 10
        self.pos_embed = build_sincos_position_embedding([10, 12, 10], self.out_dim, 3)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.out_dim)).to(device)
        # vision_encoder
        self.blocks = self.M3D_CLIP.vision_encoder.blocks
        # text_encoder
        self.text_encoder = PPMIPromptEncoder(self.M3D_CLIP, self.tokenizer, device=device)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_dim)).to(device)
        # patch dim匹配大模型
        self._fc2 = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.ReLU())

        self.layerNorm = nn.LayerNorm(self.out_dim)
        # 添加adapter层
        self.adapter = Adapter(dim=self.out_dim)
        # cls_token分类器
        self.classifier = nn.Linear(self.out_dim, self.n_classes)
        # 温度系数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        # prompt结果分类
        self.patch_mlp = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 2))

    def forward(self, x):
        # [B,N,dim=384]
        CNN_features = self.preModel(x)
        patch_text_features = self.text_encoder()
        img_izen = torch.nn.functional.normalize(torch.matmul(CNN_features, self.i_izen), dim=-1)
        pt_izen = torch.nn.functional.normalize(torch.matmul(patch_text_features, self.t_izen), dim=-1)
        pt_izen = pt_izen.squeeze(dim=0)
        logit_scale = self.logit_scale.exp()
        similarity_patch = logit_scale * img_izen @ pt_izen.T
        patch_logits = torch.softmax(similarity_patch, dim=-1)
        sum_logit = patch_logits.sum(dim=-1)
        sum_logit = torch.squeeze(sum_logit)
        c1 = patch_logits[:, :, :4]
        c2 = patch_logits[:, :, 4:]
        # c3 = patch_logits[:, :, 8:12]
        sum1 = c1.sum(dim=-1)
        sum2 = c2.sum(dim=-1)
        # sum3 = c3.sum(dim=-1)
        # 获取每个张量中最大的值,每个patch中的最大值，最相关的是哪一个类别
        max_values = torch.max(sum1, sum2)
        # max_values = torch.max(max_values, sum3)
        final_tensor = max_values
        # 归一化权重
        saigo = torch.div(final_tensor.to(self.device), sum_logit.to(self.device))
        # 权重放大
        saigo = torch.mul(saigo, 1)
        saigo = saigo.unsqueeze(dim=2)
        saigo = saigo.repeat(1, 1, 512)
        img_token = torch.mul(CNN_features, saigo)
        # lossAttention
        # gamma = CNN_features.size(1)
        # out_lb, f, alpha = self.attention_layer(img_token, self._fc1.weight, self._fc1.bias, gamma)
        # out_lb = out_lb.transpose(2, 1).sum(2)
        # out_lb = out_lb.view(out_lb.size(0), -1)
        # out_lb = self._fc1(out_lb)
        # patch_features
        out_ft = self._fc2(img_token)
        B = out_ft.shape[0]
        pos_embed = torch.cat((self.cls_pos_embed, self.pos_embed), dim=1)
        out_ts = torch.cat((self.cls_token.expand(B, -1, -1), out_ft), dim=1)
        out_ts = out_ts + pos_embed
        # 添加位置信息
        for block in self.blocks:
            out_ts = block(out_ts)

        # similarity_patch_logits = self.patch_mlp(similarity_patch)
        img_ft = out_ts[:, 0]
        img_ft = self.adapter(img_ft)
        # 画出一个直接分类的分支
        cls_token_nm = self.layerNorm(img_ft)
        # 空间对齐
        Img_i = torch.nn.functional.normalize(torch.matmul(img_ft, self.W_Img), dim=-1)
        patch_i = torch.nn.functional.normalize(torch.matmul(patch_text_features, self.W_Text), dim=-1)
        Img_i = torch.squeeze(Img_i, dim=0)
        patch_i = torch.squeeze(patch_i, dim=0)
        # clip分类
        clip_logits = torch.matmul(Img_i, patch_i.T) * logit_scale
        # cls_token = self.adpater(cls_token)
        clip_logits = self.patch_mlp(clip_logits)
        # similarity_logits = similarity_cls
        # Y_hat = torch.argmax(cls_logits, dim=1)
        # Y_prob = F.softmax(cls_logits, dim=1)
        cls_logits = self.classifier(cls_token_nm)

        return clip_logits, cls_logits


if __name__ == "__main__":
    # from weight_loss import CrossEntropyLoss as CE

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
    clip_logits,cls_logits = model(data)
    criterion1 = torch.nn.CrossEntropyLoss(
        reduction='none', size_average=True).to(device)
    loss_1 = criterion1(clip_logits, label)

    # loss_patch = criterion1(similarity_patch_logits, label)
    loss = loss_1
    print(loss)
    Y_hat = torch.argmax(clip_logits, dim=1)
    Y_prob = F.softmax(clip_logits, dim=1)
    print(cls_logits, Y_hat, Y_prob)
    # print(results_dict)
    # Modeleval()
    # from torchsummary import summary

    # summary(model, input_size=((1, 160, 192, 160)))
    # Modeleval()
