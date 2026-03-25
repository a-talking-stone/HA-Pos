# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon

cv2.setNumThreads(0)
    
class DatasetNotFoundError(Exception):
    pass

class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
                albumentations.Blur(p=0.01),
                albumentations.MedianBlur(p=0.01),
                albumentations.ToGray(p=0.01),
                albumentations.CLAHE(p=0.01),
                albumentations.RandomBrightnessContrast(p=0.0),
                albumentations.RandomGamma(p=0.0),
                albumentations.ImageCompression(quality_lower=75, p=0.0)])
    
    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh,imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3])/2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']
        #self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1-y
            
        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1-x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh==imgh, new_imgw==imgw
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Crop image
        iscropped=False
        if random.random() < 0.5:
            left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
            if left >= new_imgw/2:
                start_cropped_x = random.randint(0, int(0.15*new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw/2:
                start_cropped_x = random.randint(int(0.85*new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh/2:
                start_cropped_y = random.randint(0, int(0.15*new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh/2:
                start_cropped_y = random.randint(int(0.85*new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left/cropped_imgw, top/cropped_imgh, right/cropped_imgw, bottom/cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left*new_cropped_imgw, top*new_cropped_imgh, right*new_cropped_imgw, bottom*new_cropped_imgh 
            x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
            iscropped=True
        #if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        #print(type(img))
        #draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        #print(('draw_bbox', iscropped, draw_bbox), flush=True)
        #img_new=draw_rectangle(img, draw_bbox)
        #cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x-w/2), y-h/2, x+w/2, y+h/2]
        #print(bbox)
        #print(new_bbox)
        #print('---end---')
        return img, np.array(new_bbox, dtype=int)

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

# def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
#     """Visualizes a single bounding box on the image"""
#     x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
#     print(bbox, flush=True)
#
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
#
#     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
#     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#     cv2.putText(
#         img,
#         text=class_name,
#         org=(x_min, y_min - int(0.3 * text_height)),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.35,
#         color=TEXT_COLOR,
#         lineType=cv2.LINE_AA,
#     )
#     return img


def generate_heatmap_overlay(img, attn_score, alpha=0.5):
    """生成叠加在原图上的热图，高注意力区域为红色"""
    # attn_score: (H, W) 的注意力分数
    # img: (H, W, 3) 的 BGR 图像
    attn = attn_score.cpu().detach().numpy()  # 转换为 NumPy 数组
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)  # 归一化到 [0, 1]

    # 上采样到原始图像大小
    heatmap = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * heatmap)  # 转换为 [0, 255]

    # 应用伪彩色 (COLORMAP_JET: 低->蓝， 中->绿， 高->红)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加热图到原图
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0.0)
    return overlay


def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化的图像张量反归一化为 NumPy 数组"""
    img = img_tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)  # 从 (C, H, W) 转为 (H, W, C)
    img = img * np.array(std) + np.array(mean)  # 反归一化
    img = np.clip(img, 0, 1) * 255  # 转换为 0-255 范围
    img = img.astype(np.uint8)  # 转换为 uint8 类型
    return img


def visualize_point(img, x, y, color=(0, 0, 255), radius=5, thickness=-1):
    """在图像上绘制红点"""
    x, y = int(x), int(y)
    cv2.circle(img, (x, y), radius, color, thickness)  # 默认红色 (BGR)
    return img


def get_bbox_center(bbox):
    """从边界框计算中心点"""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return [x_center, y_center]


def visualize_bbox(img, bbox, class_name, color=(0, 0, 255), thickness=2, show_label=False):
    """在图像上绘制单个边界框"""
    # bbox 格式为 (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # 绘制边框（使用 BGR 格式）
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # 如果需要显示标签
    if show_label:
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 绘制标签背景（可选，设置为与边框颜色相同，避免黑色）
        cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)),
                      (x_min + text_width, y_min), color, -1)
        # 绘制标签文字（白色）
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.5 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),  # 白色文字
            lineType=cv2.LINE_AA,
        )
    return img

class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='train', img_size=1024,
                 transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment=augment

        self.myaugment = MyAugment()

        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 256) #52 #32
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)
        else:
            assert(False)
        
        self.rs_transform = albumentations.Compose([   
            albumentations.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.2),
	        albumentations.RandomRotate90(p=0.5),
	        albumentations.GaussNoise(p=0.5),
	        albumentations.HueSaturationValue(p=0.3),
	        albumentations.OneOf([
		        albumentations.Blur(p=0.4),
		        albumentations.MedianBlur(p=0.3),
	        ], p=0.5),
	        albumentations.OneOf([
		        albumentations.RandomBrightnessContrast(p=0.4),
		        albumentations.CLAHE(p=0.3),
	        ], p=0.5),
	        albumentations.ToGray(p=0.2),
	        albumentations.RandomGamma(p=0.3),], bbox_params=albumentations.BboxParams(format='pascal_voc',label_fields=['class_labels']))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # (131425, '131425_539733_4880095.jpg', '539827_4880079_300_32615.jpg', (383, 380), (191, 190),
        #  [258.56, 325.97333333333336, 592.7594666667779, 657.0325333340963],
        #  [495.99146666685743, 325.97333333333336, 592.7594666667779, 558.0458666674297, 355.2938666665554,
        #   657.0325333340963, 258.56, 424.96000000000004], 'building_retail')
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[idx]
        
        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        
        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)
        
        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)
        if self.augment:
            # # Apply some augmentations to query image as well
            # query_transform = albumentations.Compose([
            #     albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            #     albumentations.GaussNoise(p=0.2),
            #     albumentations.Blur(blur_limit=3, p=0.1),
            # ])
            # queryimg = query_transform(image=queryimg)['image']

            rs_transformed = self.rs_transform(image=rsimg, bboxes=[list(bbox)],class_labels=[cls_name])
            rsimg = rs_transformed['image']
            bbox = rs_transformed['bboxes'][0][0:4]

        # Norm, to tensor
        if self.transform is not None:
            rsimg = self.transform(rsimg.copy())
            queryimg = self.transform(queryimg.copy())
        
        query_featuremap_hw = self.query_featuremap_hw
        click_hw = (int(click_xy[1]), int(click_xy[0]))
        
        mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
        click_h = [pow(one-click_hw[0],2) for one in range(query_featuremap_hw[0])]
        click_w = [pow(one-click_hw[1],2) for one in range(query_featuremap_hw[1])]
        norm_hw = pow(query_featuremap_hw[0]*query_featuremap_hw[0] + query_featuremap_hw[1]*query_featuremap_hw[1], 0.5)
        for i in range(query_featuremap_hw[0]):
            for j in range(query_featuremap_hw[1]):
                tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
                mat_clickhw[i, j] = tmp_val * tmp_val
        
        return rsimg_name, queryimg, rsimg, click_xy, mat_clickhw, np.array(bbox, dtype=np.float32), idx