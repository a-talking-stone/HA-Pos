# -*- coding: utf8 -*-

import os
import sys
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2
import matplotlib.pyplot as plt
from thop import profile
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader import RSDataset, visualize_bbox, denormalize, visualize_point, get_bbox_center, generate_heatmap_overlay
from model.DetGeo import DetGeo
from model.loss import yolo_loss, build_target, adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc
from utils.checkpoint import save_checkpoint, rename_best_model, load_pretrain, load_resume

def main():
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--dataset', type=str, default='RS', help='dataset name')##
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='/home/fhr/DetGeo/data', help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='CVOGL_SVI', type=str, help='CVOGL_DroneAerial/CVOGL_SVI')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=1024
    if args.data_name == 'CVOGL_DroneAerial':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    elif args.data_name == 'CVOGL_SVI':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    else:
        assert(False)
    args.anchors = anchors

    ## save logs
    if args.savename=='default':
        args.savename = '%s_batch%d' % (args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='train',
                         img_size=args.img_size,
                         transform=input_transform,
                         augment=True)
    val_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='val',
                         img_size = args.img_size,
                         transform=input_transform)
    test_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='test',
                         img_size = args.img_size,
                         transform=input_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    
    ## Model
    model = DetGeo()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    # ========================== 计算 FLOPs ==========================
    print("\n[Info] Calculating FLOPs...")
    try:
        #切换到评估模式
        model.eval()
        
        iter_loader = iter(test_loader)
        _, query_imgs_sample, rs_imgs_sample, _, mat_clickxy_sample, _, _ = next(iter_loader)

        input_query = query_imgs_sample[0:1].cuda()
        input_rs = rs_imgs_sample[0:1].cuda()
        input_click = mat_clickxy_sample[0:1].cuda()
        
        model_ops = model.module if isinstance(model, torch.nn.DataParallel) else model
        

        flops, params = profile(model_ops, inputs=(input_query, input_rs, input_click), verbose=False)
        
        print('=============================================')
        print(f'Input Size: {args.img_size} x {args.img_size}')
        print(f'GFLOPs: {flops / 1e9 :.2f} G')
        print(f'Params: {params / 1e6 :.2f} M')
        print('=============================================\n')
        
    except Exception as e:
        print(f"[Warning] Failed to calculate FLOPs: {e}")
    # 强制将底层模型的所有 buffer 和参数移回 GPU
    if isinstance(model, torch.nn.DataParallel):
        model.module.cuda()
    else:
        model.cuda()
    
    # 确保模型处于Evaluation模式(因为thop可能会改变状态)
    model.eval() 
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    # ========================== 计算 FLOPs ==========================

    optimizer = torch.optim.RMSprop([{'params': model.parameters()},], lr=args.lr, weight_decay=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ## training and testing
    best_accu = -float('Inf')

    if args.resume:
        start_epoch, model, best_accu, optimizer = load_resume(model, optimizer, args, logging)

    if args.test:
        _ = test_epoch(test_loader, model, args)
    elif args.val:
        _ = test_epoch(val_loader, model, args)
    else:
        if args.resume:
            for epoch in range(start_epoch, args.max_epoch):
                adjust_learning_rate(args, optimizer, epoch)
                gc.collect()
                train_epoch(train_loader, model, optimizer, epoch, args)
                accu_new = test_epoch(val_loader, model, args)
                ## remember best accu and save checkpoint
                is_best = accu_new > best_accu
                best_accu = max(accu_new, best_accu)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': accu_new,
                    'optimizer': optimizer.state_dict(),
                }, is_best, epoch, args, filename=args.savename)
            # 训练结束后重命名最好模型文件
            rename_best_model(args, args.savename, best_accu)
            print('\nBest Accu: %f\n' % best_accu)
            logging.info('\nBest Accu: %f\n' % best_accu)
        else:
            for epoch in range(args.max_epoch):
                adjust_learning_rate(args, optimizer, epoch)
                gc.collect()
                train_epoch(train_loader, model, optimizer, epoch, args)
                accu_new = test_epoch(val_loader, model, args)
                ## remember best accu and save checkpoint
                is_best = accu_new > best_accu
                best_accu = max(accu_new, best_accu)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': accu_new,
                    'optimizer': optimizer.state_dict(),
                }, is_best, epoch, args, filename=args.savename)
            # 训练结束后重命名最好模型文件
            rename_best_model(args, args.savename, best_accu)
            print('\nBest Accu: %f\n' % best_accu)
            logging.info('\nBest Accu: %f\n' % best_accu)


def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_cls_losses = AverageMeter()
    avg_geo_losses = AverageMeter()
    avg_accu = AverageMeter()
    avg_accu_center = AverageMeter()
    avg_iou = AverageMeter()

    model.train()
    end = time.time()
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()
    
    for batch_idx, (rsimg_names, query_imgs, rs_imgs, click_xy, mat_clickxy, ori_gt_bbox, _) in enumerate(train_loader):
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        pred_anchor, _ = model(query_imgs, rs_imgs, mat_clickxy)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        ## convert gt box to center+offset format
        new_gt_bbox, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        # loss
        loss_geo, loss_cls = yolo_loss(pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, args.img_size)
        loss = loss_cls + loss_geo * args.beta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_losses.update(loss.item(), query_imgs.shape[0])
        avg_geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        avg_cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        
        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5])
        accu = accu_list[0]
        ## metrics
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu.update(accu, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}/{1}][{2}/{3}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Geo Loss {geo.val:.4f} ({geo.avg:.4f})\t' \
                'Cls Loss {cls.val:.4f} ({cls.avg:.4f})\t' \
                'Accu {accu.val:.4f} ({accu.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    epoch+1, args.max_epoch, batch_idx, len(train_loader), batch_time=batch_time, \
                    loss=avg_losses, geo=avg_geo_losses, cls=avg_cls_losses, accu=avg_accu, miou=avg_iou, accu_c=avg_accu_center)
            print(print_str)
            logging.info(print_str)

def test_epoch(data_loader, model, args):
    batch_time = AverageMeter()
    inference_time = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()

    # --- 为不同尺寸的物体初始化 AverageMeter ---
    # 定义尺寸区间 (平方米)，例如参考 Sun et al. 的论文
    # (0, 1000], (1000, 2000], (2000, 3000], (3000, 4000], >4000
    size_bins_m2 = [1000, 2000, 3000, 4000]
    num_size_bins = len(size_bins_m2) + 1
    avg_accu25_by_size = [AverageMeter() for _ in range(num_size_bins)]
    # ------------------------------------------

    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    #print(datetime.datetime.now())
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    for batch_idx, (rsimg_names, query_imgs, rs_imgs, click_xy, mat_clickxy, ori_gt_bbox, idx) in enumerate(data_loader):
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        torch.cuda.synchronize() # 等待当前所有GPU任务完成
        infer_start_time = time.time()

        with torch.no_grad():
            pred_anchor, attn_score = model(query_imgs, rs_imgs, mat_clickxy)
        
        torch.cuda.synchronize() # 等待模型推理相关的GPU任务完成
        infer_end_time = time.time()
        inference_time.update((infer_end_time - infer_start_time)/query_imgs.shape[0], query_imgs.size(0))

        pred_anchor = pred_anchor.view(pred_anchor.shape[0],\
            9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        _, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        accu_list, accu_center, iou, each_acc_list, pred_bbox, target_bbox = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5, 0.25])

        for i in range(rs_imgs.shape[0]):
            ground_coverage_side_m = float(rsimg_names[i].split('_')[-2])
            # 计算分辨率
            spatial_resolution_m_per_px = ground_coverage_side_m / args.img_size
            # print(f"分辨率:{spatial_resolution_m_per_px}")
            # 获取当前样本的 GT BBox (像素单位)
            gt_box_px = ori_gt_bbox[i].cpu().numpy()  # [xmin, ymin, xmax, ymax]
            box_width_px = gt_box_px[2] - gt_box_px[0]
            box_height_px = gt_box_px[3] - gt_box_px[1]
            # 计算物理面积 (平方米)
            box_width_m = box_width_px * spatial_resolution_m_per_px
            box_height_m = box_height_px * spatial_resolution_m_per_px
            object_area_m2 = box_width_m * box_height_m
            # print(f"物理面积:{object_area_m2}")
            # 判断属于哪个尺寸区间
            bin_idx = -1
            for j, upper_bound in enumerate(size_bins_m2):
                if object_area_m2 <= upper_bound:
                    bin_idx = j
                    break
            if bin_idx == -1:  # 大于所有定义的上限
                bin_idx = len(size_bins_m2)
            # print(accu_list)
            # 我们用 acc@0.25，所以是 accu_list[1]
            current_sample_acc25 = accu_list[1]
            avg_accu25_by_size[bin_idx].update(current_sample_acc25, 1)

                # if batch_idx < 20:
        #     data = args.data_name
        #     dataset = "val" if args.val else "test" if args.test else ""
        #     os.makedirs('visualizations', exist_ok=True)  # 创建保存目录
        #     vis = [3,16,18,107,141]
        #     for i in range(query_imgs.shape[0]):
        #         if data == "CVOGL_DroneAerial" and dataset == "val" and idx[i] in vis:
        #             # 反归一化图像
        #             query_img = denormalize(query_imgs[i])
        #             rs_img = denormalize(rs_imgs[i])
        #             # 转换为 BGR 格式（OpenCV 使用 BGR）
        #             query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
        #             rs_img_bgr = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)
        #             pred_bbox_i = pred_bbox[i].cpu().numpy()  # 预测边框
        #             target_bbox_i = target_bbox[i].cpu().numpy()  # 真实边框
        #
        #             # 1. 带红点的原图（使用 mat_clickxy）
        #             img_with_point = query_img_bgr.copy()
        #             visualize_point(img_with_point, click_xy[0][i], click_xy[1][i], color=(0, 0, 255), radius=10)
        #             # 构造目标目录路径
        #             save_dir = f'visualizations/{data}/{dataset}/point/all_2/可视化'
        #             # 检查目标目录是否存在，不存在则创建
        #             if not os.path.exists(save_dir):
        #                 os.makedirs(save_dir)
        #             # 保存图像
        #             cv2.imwrite(f'{save_dir}/{batch_idx}_{idx[i]}.jpg', img_with_point)
        #             #
        #             # 2.边框图:在同一张图上绘制预测框（红色）和真实框（绿色）
        #             # img_with_boxes = rs_img_bgr.copy()
        #             # # 绘制预测框 (BGR 红色: (0, 0, 255))
        #             # img_with_boxes = visualize_bbox(img_with_boxes, pred_bbox_i, "Predicted", color=(0, 0, 255),
        #             #                                 thickness=7)
        #             # # 绘制真实框 (BGR 绿色: (0, 255, 0))
        #             # img_with_boxes = visualize_bbox(img_with_boxes, target_bbox_i, "Ground Truth", color=(0, 255, 0),
        #             #                                 thickness=7)
        #             # # 构造目标目录路径
        #             # save_dir = f'visualizations/{data}/{dataset}/box/baseline/可视化'
        #             # # 检查目标目录是否存在，不存在则创建
        #             # if not os.path.exists(save_dir):
        #             #     os.makedirs(save_dir)
        #             # # 保存图像
        #             # cv2.imwrite(f'{save_dir}/{batch_idx}_{idx[i]}.jpg', img_with_boxes)
        #
        #             # 3. 热图
        #             img_with_heatmap = rs_img_bgr.copy()
        #             # img_with_heatmap = generate_heatmap_overlay(img_with_heatmap, attn_score[i], center_point, alpha=0.5, sigma=20)
        #             img_with_heatmap = generate_heatmap_overlay(img_with_heatmap, attn_score[i])
        #             # 构造目标目录路径
        #             save_dir = f'visualizations/{data}/{dataset}/heatmap/all_2/可视化'
        #             # 检查目标目录是否存在，不存在则创建
        #             if not os.path.exists(save_dir):
        #                 os.makedirs(save_dir)
        #             # 保存图像
        #             cv2.imwrite(f'{save_dir}/{batch_idx}_{idx[i]}.jpg', img_with_heatmap)

        avg_accu50.update(accu_list[0], query_imgs.shape[0])
        avg_accu25.update(accu_list[1], query_imgs.shape[0])
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Inference_Time {infer_time.val:.4f}s/img ({infer_time.avg:.4f}s/img)\t' \
                'Accu50 {accu50.val:.4f} ({accu50.avg:.4f})\t' \
                'Accu25 {accu25.val:.4f} ({accu25.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(data_loader), batch_time=batch_time, \
                    infer_time=inference_time,
                    accu50=avg_accu50, accu25=avg_accu25, miou=avg_iou, accu_c=avg_accu_center)
            print(print_str)
            logging.info(print_str)

    # --- 打印和记录按尺寸划分的 acc@0.25 ---
      # 最终测试结束后，可以打印或记录总的平均推理时间
    final_avg_inference_time = inference_time.avg
    print(f"\nAverage Inference Time per item: {final_avg_inference_time * 1000:.2f} ms")
    logging.info(f"Average Inference Time per item: {final_avg_inference_time:.6f} s")
    
    print_str_size = "Acc@0.25 by Object Size (m^2):"
    logging.info(print_str_size)
    print(print_str_size)

    size_bin_labels = []
    lower_bound = 0
    for upper_bound in size_bins_m2:
        size_bin_labels.append(f"({lower_bound}, {upper_bound}]")
        lower_bound = upper_bound
    size_bin_labels.append(f"> {size_bins_m2[-1]}")

    for i in range(num_size_bins):
        acc_val = avg_accu25_by_size[i].avg if avg_accu25_by_size[i].count > 0 else -1  # 用 -1 表示无样本
        count = avg_accu25_by_size[i].count
        size_label_str = f"  {size_bin_labels[i]}: {acc_val:.4f} (count: {count})"
        print(size_label_str)
        logging.info(size_label_str)
    # ----------------------------------------

    print(avg_accu50.avg, avg_accu25.avg, avg_iou.avg, avg_accu_center.avg)
    logging.info("%f, %f, %f, %f" % (avg_accu50.avg, avg_accu25.avg, float(avg_iou.avg), avg_accu_center.avg))
    if args.val:
        logging.info("val结束")
    elif args.test:
        logging.info("test结束\n")
    return avg_accu50.avg


if __name__ == "__main__":
    main()
