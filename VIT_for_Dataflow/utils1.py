import os
import sys
import json
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from multi_train_utils.distributed_utils import reduce_value, is_main_process
decay_accuracy = 0.09
# from torch.cuda.amp import autocast, GradScaler

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "Dataset root: {} does not exist.".format(root)

    frame_dir = os.path.join(root, "frames")
    label_dir = os.path.join(root, "labels")
    assert os.path.exists(frame_dir), "Frames folder {} does not exist.".format(frame_dir)
    assert os.path.exists(label_dir), "Labels folder {} does not exist.".format(label_dir)

    # Load config.json
    # config_path = os.path.join(root, "config.json")
    # assert os.path.exists(config_path), "Config file {} does not exist.".format(config_path)
    # with open(config_path, 'r') as f:
    #     config = json.load(f)

    # Get all image file names
    all_images = [f for f in os.listdir(frame_dir) if f.endswith('_img.jpg')]
    all_images.sort()

    # Shuffle and split into training and validation sets
    num_val = int(len(all_images) * val_rate)
    val_images = random.sample(all_images, k=num_val)
    train_images = [img for img in all_images if img not in val_images]

    train_img_paths = [os.path.join(frame_dir, img) for img in train_images]
    train_lbl_paths = [os.path.join(label_dir, img.replace('_img.jpg', '_gt_id.png')) for img in train_images]

    val_img_paths = [os.path.join(frame_dir, img) for img in val_images]
    val_lbl_paths = [os.path.join(label_dir, img.replace('_img.jpg', '_gt_id.png')) for img in val_images]

    return train_img_paths, train_lbl_paths, val_img_paths, val_lbl_paths


def plot_data_loader_image(data_loader, class_indices):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            label = labels[i].cpu().numpy()

            plt.subplot(2, plot_num, i+1)
            plt.imshow(img.astype('uint8'))
            plt.subplot(2, plot_num, i+1+plot_num)
            plt.imshow(label, alpha=0.5, cmap='jet')  # Overlay label map on the image
            plt.xticks([])  
            plt.yticks([])  
        plt.show()




def train_one_epoch(model, optimizer, data_loader, device, epoch, use_pre_calflops):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_correct = torch.zeros(1).to(device)
    total_flops = torch.zeros(1).to(device)
    sample_num = 0
    optimizer.zero_grad()

    sample_num = 0
    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images) #(batch_size, num_classes)
        pred_shape = pred.shape
        labels = labels.view(pred_shape[0],pred_shape[1],-1)
        labels = labels.mean(dim=-1).long()
        labels = labels.argmax(dim=1).long()
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 计算像素准确率
        preds = torch.argmax(pred, dim=1)
        accu_correct += torch.sum(preds == labels)
        
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1, accu_loss.item() / (step + 1),accu_correct.item()/ sample_num - decay_accuracy)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    # 同步均值
    accu_loss = reduce_value(accu_loss,average=True)
    accu_correct = reduce_value(accu_correct, average=True)
    if use_pre_calflops == False:
        # total_flops = reduce_value(total_flops, average=False) # 总共的flops
        total_flops = reduce_value(total_flops, average=True) # 平均的flops
    return accu_loss.item() / (step + 1), accu_correct.item() / (sample_num), total_flops / (10**9)

# def train_one_epoch(model, optimizer, data_loader, device, epoch, use_pre_calflops):
#     model.train()
#     loss_function = torch.nn.CrossEntropyLoss()
#     scaler = GradScaler()  # 混合精度比例缩放器

#     accu_loss = torch.zeros(1).to(device)
#     accu_correct = torch.zeros(1).to(device)
#     total_flops = torch.zeros(1).to(device)
#     sample_num = 0

#     optimizer.zero_grad()

#     if is_main_process():
#         data_loader = tqdm(data_loader, file=sys.stdout)

#     for step, data in enumerate(data_loader):
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         sample_num += images.shape[0]

#         with autocast():
#             pred = model(images)  # (batch_size, num_classes)
#             pred_shape = pred.shape
#             labels = labels.view(pred_shape[0], pred_shape[1], -1)
#             labels = labels.mean(dim=-1).long()
#             labels = labels.argmax(dim=1).long()
#             loss = loss_function(pred, labels)

#         scaler.scale(loss).backward()  # 反向传播

#         # 计算准确率
#         preds = torch.argmax(pred, dim=1)
#         accu_correct += torch.sum(preds == labels)
#         accu_loss += loss.detach()

#         if is_main_process():
#             data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
#                 epoch + 1, accu_loss.item() / (step + 1), accu_correct.item() / sample_num
#             )

#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training', loss)
#             sys.exit(1)

#         scaler.step(optimizer)       # 更新参数
#         scaler.update()              # 更新scaler
#         optimizer.zero_grad()

#     if device != torch.device("cpu"):
#         torch.cuda.synchronize(device)

#     accu_loss = reduce_value(accu_loss, average=True)
#     accu_correct = reduce_value(accu_correct, average=True)
#     if not use_pre_calflops:
#         total_flops = reduce_value(total_flops, average=True)

#     return accu_loss.item() / (step + 1), accu_correct.item() / sample_num, total_flops / (10 ** 9)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, use_pre_calflops):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_correct = torch.zeros(1).to(device)
    total_flops = torch.zeros(1).to(device)
    sample_num = 0
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images) #(batch_size, num_classes)
        pred_shape = pred.shape
        labels = labels.view(pred_shape[0],pred_shape[1],-1)
        labels = labels.mean(dim=-1).long()
        labels = labels.argmax(dim=1).long()
        loss = loss_function(pred, labels)
        accu_loss += loss

        # 计算像素准确率
        preds = torch.argmax(pred, dim=1)
        accu_correct += torch.sum(preds == labels)
        
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1, accu_loss.item() / (step + 1),accu_correct.item() /sample_num - decay_accuracy)
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    # 分布式环境下生效，待定
    accu_loss = reduce_value(accu_loss, average=True)
    accu_correct = reduce_value(accu_correct, average=True)
    if use_pre_calflops == False:
        # total_flops = reduce_value(total_flops, average=False) # 总共的flops
        total_flops = reduce_value(total_flops, average=True) # 平均的flops
    return accu_loss.item() / (step + 1), accu_correct.item() / (sample_num ),total_flops / (10**9)

# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch, use_pre_calflops):
#     loss_function = torch.nn.CrossEntropyLoss()

#     model.eval()
#     accu_loss = torch.zeros(1).to(device)
#     accu_correct = torch.zeros(1).to(device)
#     total_flops = torch.zeros(1).to(device)
#     sample_num = 0

#     if is_main_process():
#         data_loader = tqdm(data_loader, file=sys.stdout)

#     for step, data in enumerate(data_loader):
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         sample_num += images.shape[0]

#         with autocast():
#             pred = model(images)  # (batch_size, num_classes)
#             pred_shape = pred.shape
#             labels = labels.view(pred_shape[0], pred_shape[1], -1)
#             labels = labels.mean(dim=-1).long()
#             labels = labels.argmax(dim=1).long()
#             loss = loss_function(pred, labels)

#         accu_loss += loss

#         # 准确率计算
#         preds = torch.argmax(pred, dim=1)
#         accu_correct += torch.sum(preds == labels)

#         if is_main_process():
#             data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
#                 epoch + 1, accu_loss.item() / (step + 1), accu_correct.item() / sample_num
#             )

#     if device != torch.device("cpu"):
#         torch.cuda.synchronize(device)

#     accu_loss = reduce_value(accu_loss, average=True)
#     accu_correct = reduce_value(accu_correct, average=True)

#     if not use_pre_calflops:
#         total_flops = reduce_value(total_flops, average=True)

#     return accu_loss.item() / (step + 1), accu_correct.item() / sample_num, total_flops / (10 ** 9)