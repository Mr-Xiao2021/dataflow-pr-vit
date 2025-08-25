import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
# from calflops import calculate_flops
import matplotlib.pyplot as plt
from multi_train_utils.distributed_utils import reduce_value, is_main_process
from typing import List, Tuple


def read_split_data(root: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    assert os.path.exists(train_dir), f"train dir: {train_dir} does not exist."
    assert os.path.exists(val_dir), f"val dir: {val_dir} does not exist."

    # 提取所有类别（子文件夹名）并建立索引
    classes = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    classes.sort()
    class_indices = {k: v for v, k in enumerate(classes)}
    
    # 保存类别索引映射
    with open("class_indices.json", "w") as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=4)

    # 支持的图像文件后缀
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    def get_images_and_labels(directory: str):
        images, labels = [], []
        for cla in classes:
            cla_path = os.path.join(directory, cla)
            if not os.path.isdir(cla_path):
                continue
            for file in sorted(os.listdir(cla_path)):
                if os.path.splitext(file)[-1] in supported:
                    images.append(os.path.join(cla_path, file))
                    labels.append(class_indices[cla])
        return images, labels

    train_images_path, train_images_label = get_images_and_labels(train_dir)
    val_images_path, val_images_label = get_images_and_labels(val_dir)

    print(f"Found {len(train_images_path)} training images.")
    print(f"Found {len(val_images_path)} validation images.")
    assert len(train_images_path) > 0, "Training set is empty!"
    assert len(val_images_path) > 0, "Validation set is empty!"

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_pre_calflops):
    model.train() #开启训练模式
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    total_flops = torch.zeros(1).to(device) # 浮点运算数
    optimizer.zero_grad()

    sample_num = 0
    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # FLOPs 计算 - 每次前向传播时累积每个batch的FLOPs
        # if use_pre_calflops == False:
        #     flops, _, _ = calculate_flops(model, input_shape=tuple(images.shape),
        #                                   print_results=False,print_detailed=False, output_as_string=False)
        #     # total_flops += flops
        #     total_flops[0] = torch.tensor(flops).to(device)

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()

        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    # 同步均值
    accu_loss = reduce_value(accu_loss,average=True)
    accu_num = reduce_value(accu_num, average=True)
    if use_pre_calflops == False:
        # total_flops = reduce_value(total_flops, average=False) # 总共的flops
        total_flops = reduce_value(total_flops, average=True) # 总共的flops
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, total_flops / (10**9)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, use_pre_calflops):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_flops = torch.zeros(1).to(device) # 浮点运算数

    sample_num = 0
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        if use_pre_calflops == False:
            flops, _, _ = calculate_flops(model, input_shape=tuple(images.shape),
                                          print_results=False,print_detailed=False, output_as_string=False)
            # total_flops += flops
            total_flops[0] = torch.tensor(flops).to(device)

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        # 正确样本的个数
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        
    
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    # 分布式环境下生效，待定
    accu_loss = reduce_value(accu_loss, average=True)
    accu_num = reduce_value(accu_num, average=True)
    if use_pre_calflops == False:
        # total_flops = reduce_value(total_flops, average=False) # 总共的flops
        total_flops = reduce_value(total_flops, average=True) # 总共的flops
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,total_flops / (10**9)
    