# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

from torchvision.ops import DeformConv2d

from model.darknet import *
import torchvision.models as models
from einops.layers.torch import Rearrange

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        #print(('x1', x.shape), flush=True)
        x = self.base_model.layer2(x)
        #print(('x2', x.shape), flush=True)
        x = self.base_model.layer3(x)
        #print(('x3', x.shape), flush=True)
        x = self.base_model.layer4(x)
        #print(('x4', x.shape), flush=True)
        return x

        # output = []
        # # ('x1', torch.Size([6, 64, 64, 64]))
        # # ('x2', torch.Size([6, 128, 32, 32]))
        # # ('x3', torch.Size([6, 256, 16, 16]))
        # # ('x4', torch.Size([6, 512, 8, 8]))
        # x = self.base_model.layer1(x)
        # output.append(x)
        # # print(('x1', x.shape), flush=True)
        # x = self.base_model.layer2(x)
        # output.append(x)
        # # print(('x2', x.shape), flush=True)
        # x = self.base_model.layer3(x)
        # output.append(x)
        # # print(('x3', x.shape), flush=True)
        # x = self.base_model.layer4(x)
        # output.append(x)
        # # print(('x4', x.shape), flush=True)
        # return output

class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = Variable(torch.zeros(B, H*W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn


class DeformConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=nn.SiLU()):
        super(DeformConvBNAct, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Deformable Convolution
        self.deform_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        # Offset 和 Modulation 的预测网络
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 每个采样点预测 2D 偏移 (x, y)
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.modulation_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,  # 每个采样点预测调制权重
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation if activation is not None else nn.Identity()

        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.modulation_conv.weight, 0)
        if self.modulation_conv.bias is not None:
            nn.init.constant_(self.modulation_conv.bias, 1)  # 默认调制权重为 1

    def forward(self, x):
        # 预测偏移量和调制权重
        offset = self.offset_conv(x)  # (B, 2*k*k, H, W)
        modulation = torch.sigmoid(self.modulation_conv(x))  # (B, k*k, H, W)，值范围 [0, 1]

        # 执行 DCN v2
        x = self.deform_conv(x, offset=offset, mask=modulation)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, activation=nn.SiLU()):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Squeeze-and-Excitation block.
        Args:
            channel (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate layer.
        """
        super(SELayer, self).__init__()
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c) # (B, C)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1) # (B, C, 1, 1)
        # Scale original features
        return x * y.expand_as(x)

# --- Improved Decoupled Detection Head ---
class DecoupledDetHead(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, num_anchors=9, num_classes=1, activation=nn.SiLU()):
        """
        Improved Decoupled Detection Head.

        Args:
            in_channels (int): Number of input channels (from the fused feature map).
            hidden_channels (int): Number of channels in the intermediate layers.
            num_anchors (int): Number of anchors per grid cell.
            num_classes (int): Number of object classes. Set to 1 for just objectness/confidence,
                               or >1 if predicting specific classes.
            activation (nn.Module): Activation function to use.
        """
        super().__init__()
        # 控制开关
        self.use_dcn = True
        self.use_decoupled = True

        # self.se = SELayer(hidden_channels)  # Apply SE on hidden_channels

        # self.dropout = nn.Dropout2d(p=0.1)  # Use Dropout2d for conv features

        self.num_anchors = num_anchors
        self.num_classes = num_classes  # Typically 1 for objectness + Box Regression

        # Shared convolutional layers for initial refinement
        if self.use_dcn:
            self.stem = DeformConvBNAct(in_channels, hidden_channels, kernel_size=3, activation=activation)
            self.refine_conv = DeformConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        else:
            self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=3, activation=activation)
            self.refine_conv = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)


        # Classification/Objectness Branch
        self.cls_conv1 = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        self.cls_conv2 = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        # Final prediction layer for objectness/class scores
        self.cls_pred = nn.Conv2d(hidden_channels, num_anchors * self.num_classes, kernel_size=1)

        # Regression Branch
        self.reg_conv1 = ConvBNAct (hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        self.reg_conv2 = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        # Final prediction layer for bounding box regression (4 values: tx, ty, tw, th)
        self.reg_pred = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=1)

        self._initialize_biases()


        self.coupling = torch.nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation),
            nn.Conv2d(hidden_channels, 9 * 5, kernel_size=1))

    def _initialize_biases(self):
        # Optional: Initialize biases for the final prediction layers
        # Helps convergence, especially for objectness
        # Adjust the bias initialization constant `b` as needed
        for m in [self.cls_pred]:  # Potentially add self.reg_pred if needed
            if m.bias is not None:
                # Prior probability of objectness
                prior_prob = 0.01
                b = -math.log((1.0 - prior_prob) / prior_prob)
                nn.init.constant_(m.bias, b)

    def forward(self, x):
        # Input shape: (B, C_in, H, W)
        # Shared refinement
        stem_feat = self.stem(x)
        refined_feat = self.refine_conv(stem_feat)  # (B, C_hidden, H, W)

        # if self.use_se:
        #     refined_feat = self.se(refined_feat)

        # refined_feat = self.dropout(refined_feat)

        if self.use_decoupled:
            # Classification/Objectness Branch
            cls_feat = self.cls_conv1(refined_feat)
            cls_feat = self.cls_conv2(cls_feat)
            cls_output = self.cls_pred(cls_feat)  # (B, A*Nc, H, W)

            # Regression Branch
            reg_feat = self.reg_conv1(refined_feat)
            reg_feat = self.reg_conv2(reg_feat)
            reg_output = self.reg_pred(reg_feat)  # (B, A*4, H, W)

            # Reshape outputs (optional but common for loss calculation)
            # Example reshape for YOLO-style loss: (B, A, H, W, Nc) and (B, A, H, W, 4)
            B, _, H, W = x.shape
            cls_output = cls_output.view(B, self.num_anchors, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            reg_output = reg_output.view(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()

            output = torch.cat(
                (reg_output.permute(0, 1, 4, 2, 3),  # (B, A, 4, H, W)
                 cls_output.permute(0, 1, 4, 2, 3)),  # (B, A, Nc, H, W) -> (B, A, 1, H, W) if Nc=1
                dim=2  # Concatenate along the 'attributes' dimension
            )
            B, A, Nc, H, W = output.shape
            output = output.view(B, A * Nc, H, W)
        else:
            output = self.coupling(refined_feat)

        return output

class HAPos(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(HAPos, self).__init__()
        # 控制开关
        self.use_sp = True
        self.use_gcad = True

        self.heatmap_attention = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1之间的注意力权重
        )

            
        # Visual model
        self.query_resnet = MyResnet()
        self.reference_darknet = Darknet(config_path='/home/fhr/HAPos/model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('/home/fhr/HAPos/saved_models/yolov3.weights')

        # model_name = 'convnext_base.fb_in22k_ft_in1k_384'
        # # model_name = 'convnext_tiny.fb_in22k_ft_in1k_384'
        # # model_name = 'vit_tiny_patch16_384.augreg_in21k_ft_in1k'
        # self.model = timm.create_model(model_name, pretrained=False, num_classes=0, features_only=True)
        # state_dict = torch.load("./saved_models/convnext_base")
        # self.model.load_state_dict(state_dict, strict=False)
        
        use_instnorm=False

        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()

        self.query_visudim = 512
        self.reference_visudim = 512
        # self.query_visudim = 768
        # self.reference_visudim = 768

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        ## output head
        if self.use_gcad:
            self.fcn_out = DecoupledDetHead(
            in_channels=emb_size,  # Input from fused features
            hidden_channels=emb_size // 2,  # Or keep emb_size, e.g., 256 or 512
            num_anchors=9,  # Match your anchor setup
            num_classes=1  # Assuming objectness only first
        )
        else:
            self.fcn_out = torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                    nn.Conv2d(emb_size//2, 9*5, kernel_size=1))

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)
        
        query_imgs = self.combine_clickptns_conv( torch.cat((query_imgs, mat_clickptns), dim=1) )
        query_fvisu = self.query_resnet(query_imgs)

        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]
        # query_fvisu = self.model(query_imgs)[-1]
        # reference_fvisu = self.model(reference_imgs)[-1]

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape

        if self.use_sp:
            _, _, W, H = query_fvisu.shape
            # 生成热图注意力掩码（与ViT特征分辨率对齐）
            heatmap_mask = self.heatmap_attention(mat_clickptns)
            heatmap_mask = F.interpolate(heatmap_mask, size=(W, H), mode='bilinear')  # ViT输出特征尺度
            # 应用热图注意力掩码（增强查询点区域）
            query_fvisu = query_fvisu * heatmap_mask  # 逐元素相乘，抑制非关键区域

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        # cross-view fusion
        query_gvisu = torch.mean(query_fvisu.view(B, D, Hquery*Wquery), dim=2, keepdims=False).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)

        # fused_features, attn_score = self.crossview_fusionmodule(query_fvisu, reference_fvisu)

        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)

        return outbox, attn_score