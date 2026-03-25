# -*- coding:utf8 -*-

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F


def save_checkpoint(state, is_best, epoch, args, filename='default'):
    if filename == 'default':
        filename = 'filmconv_nofpn32_%s_batch%d' % (args.dataset, args.batch_size)

    checkpoint_name = '/home/fhr/DetGeo-整合/saved_models/%s_checkpoint.pth.tar' % (filename)
    best_name = '/home/fhr/DetGeo-整合/saved_models/%s_model_best.pth.tar' % (filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)


def rename_best_model(args, filename, best_accu):
    # 原始最好模型文件名
    original_best_name = '/home/fhr/DetGeo-整合/saved_models/%s_model_best.pth.tar' % (filename)
    # 新的最好模型文件名，包含 best_accu
    new_best_name = '/home/fhr/DetGeo-整合/saved_models/%s_model_best_%.6f.pth.tar' % (filename, best_accu)

    # 检查文件是否存在
    if os.path.exists(original_best_name):
        # 重命名文件
        os.rename(original_best_name, new_best_name)
        print(f"Renamed best model to: {new_best_name}")
    else:
        print(f"Best model file not found: {original_best_name}")

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        logging.info("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model


def load_resume(model, optimizer, args, logging):
    # 初始化默认值
    start_epoch = 0
    best_loss = float('inf')  # 或者根据你的需求设置为0.0
    
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(("=> loaded checkpoint (epoch {}) Loss{}"
               .format(checkpoint['epoch'], best_loss)))
        logging.info("=> loaded checkpoint (epoch {}) Loss{}"
                     .format(checkpoint['epoch'], best_loss))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        logging.info(("=> no checkpoint found at '{}'".format(args.resume)))

    return start_epoch, model, best_loss, optimizer