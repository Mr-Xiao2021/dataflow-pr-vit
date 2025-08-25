import os
import math
import tempfile
import argparse
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision import models
from my_dataset1 import MySegmentationDataset
from utils1 import read_split_data, train_one_epoch, evaluate
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from vit_model import vit_base_patch16_224_in21k as create_model

"""
单机单卡
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env train1_multi_gpu.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port='12346'  --use_env train1_multi_gpu.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='12347'  --use_env train1_multi_gpu.py
python -m torch.distributed.launch --nproc_per_node=8 --use_env train1_multi_gpu.py 
多机多卡
export NCCL_SOCKET_IFNAME=ens8f1 
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='11.11.11.15' --master_port='12345' train1_multi_gpu.py

"""

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""
    # 如果为否，那么每次运算都会计算FLOPs，将会拖慢整体训练速度；包括了train和val的，和batch_size有关
    use_pre_calflops = args.use_pre_calflops
    pre_calflops = args.pre_calflops

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        compile_start = time.time()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")


    train_images_path, train_labels_path, val_images_path, val_labels_path = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),# resnet: 256
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),# resnet: 256
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    if rank == 0:
        total_start_time = time.time()
    # 实例化训练数据集
    train_dataset = MySegmentationDataset(images_path=train_images_path,
                                          labels_path=train_labels_path,
                                          num_classes = args.num_classes,
                                          transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MySegmentationDataset(images_path=val_images_path,
                                        labels_path=val_labels_path,
                                        num_classes = args.num_classes,
                                        transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    # batch_x, batch_y
    # resnet: torch.Size([8, 3, 256, 256]) torch.Size([8, 1080, 1920])
    # ViT:    torch.Size([8, 3, 224, 224]) torch.Size([8, 13, 224, 224])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True, #将数据加载到GPU当中
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sampler, # valid_sampler直接传
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)                                    

    # 实例化模型
    # model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes).to(device) # resnet: 256
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        # del_keys = ['classifier.4.weight', 'classifier.4.bias','head.weight',"head.bias"] # resnet: 256
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            # del weights_dict[k]
            weights_dict.pop(k, None)
        print(model.load_state_dict(weights_dict, strict=False))
    else: # 检查点
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 是否冻结权重
    if args.freeze_layers:
        for name, param in model.named_parameters():
            # if "classifier" not in name: # resnet: 256
            if "head" not in name and "pre_logits" not in name:
                param.requires_grad = False
            else:
                print("Training {}".format(name))
    
    # 转为DDP模型

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if rank == 0:
        cal_time_list = []
        cal_flops_list = []
        compile_end = time.time()
        print(f'=============== CompileTime: {compile_end - compile_start:.2f} second')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            epoch_start_time = time.time()
        # train
        train_loss, train_acc,total_train_flops = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch,
                                     use_pre_calflops=use_pre_calflops)

        scheduler.step()

        val_loss, val_acc,total_val_flops = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            use_pre_calflops=use_pre_calflops)
        if rank == 0:
            epoch_end_time = time.time()

            pre_cal_flops = 1240889664
            total_samples = 5000
            print(f"{len(train_images_path) + len(val_images_path)}")
            cal_time = epoch_end_time - epoch_start_time
            cal_flops = pre_cal_flops * total_samples / cal_time / 1e9
            cal_time_list.append(cal_time)
            cal_flops_list.append(cal_flops)

            print(f"EPOCH{epoch+1} GFLOPS: {cal_flops:.2f} GFLOPS, TIME: {cal_time:.2f} seconds.")

            # save weights
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
        total_end_time = time.time()
        print("======================== DataFlow ======================== ")
        print("\t\tTotal E2E time: {:.2f} seconds".format(total_end_time - total_start_time))
        # 计算最后三轮平均Eval时间和Eval GFLOPS
        cal_time_list = cal_time_list[-3:]
        cal_flops_list = cal_flops_list[-3:]
        avg_cal_time = sum(cal_time_list) / len(cal_time_list)
        avg_cal_flops = sum(cal_flops_list) / len(cal_flops_list)
        print(f'\t\t Average Time Per Epoch: {avg_cal_time:.2f} seconds, \n\t\tAVG Calculate GFLOPS: {avg_cal_flops:.2f} GFLOPS')
        print("======================== DataFlow ======================== ")
    # 释放进程组
    cleanup()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=13)  # 增加背景类
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    # 是否使用预求的FLOPs
    parser.add_argument('--use-pre-calflops', type=bool, default=True)
    # 预求得FLOPs，可以通过设置--use-pre-calflops为True进行重新计算再更新
    parser.add_argument('--pre-calflops', type=float, default=1310.21, help='GFLOPs')

    parser.add_argument('--data-path', type=str, default="/gemini/code/DriveSeg")
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int)  # 增加local_rank
    opt = parser.parse_args()
    main(opt)