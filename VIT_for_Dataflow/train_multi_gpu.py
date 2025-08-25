import os
import math
import tempfile
import argparse
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from vit_model import vit_base_patch16_224_in21k as create_model
# from vit_model import vit_base_patch32_224_in21k as create_model
# from vit_model import vit_large_patch32_224_in21k as create_model

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup


gflops_dict = {
    "vit_b_16_224": 1240889664,
    'vit_b_32_224': 501553920,
    "vit_l_32_224": 2104470528,
}

"""
conda activate pytorch-multi-GPU-training-tutorial
单机单卡
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port='12345' --use_env train_multi_gpu.py
单机双卡
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port='12346'  --use_env train_multi_gpu.py
单机四卡
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='12347'  --use_env train_multi_gpu.py

单机八卡
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpu.py 
多机多卡：
export NCCL_SOCKET_IFNAME=ens8f1 
- 双机多卡
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='11.11.11.15' --master_port='12345' train_multi_gpu.py
- 三机多卡
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 3 --node_rank 0 --master_addr='11.11.11.15' --master_port='12345' train_multi_gpu.py
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


    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        compile_start = time.time()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    if rank == 0:
        total_start_time = time.time()
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
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
    #  batch_x, batch_y
    #  torch.Size([32, 3, 224, 224]) torch.Size([32])
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
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                if rank == 0:
                    print("training {}".format(name))
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
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
        if rank == 0:
            epoch_train_time = time.time()

        # validate
        val_loss, val_acc, total_val_flops = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     use_pre_calflops=use_pre_calflops)

        if rank == 0:
            epoch_end_time = time.time()
        
            pre_cal_flops = 1240889664
            total_samples = len(train_images_path) + len(val_images_path)

            total_flops = pre_cal_flops if use_pre_calflops else (total_train_flops + total_val_flops).item()
            
            cal_time = epoch_end_time - epoch_start_time
            cal_flops = pre_cal_flops * total_samples / cal_time / 1e9
            cal_time_list.append(cal_time)
            cal_flops_list.append(cal_flops)

            print(f"EPOCH{epoch} GFLOPS: {cal_flops:.2f} GFLOPS, TIME: {cal_time:.2f} seconds.")
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
        print(f'\t\t Average Time Per Epoch: {avg_cal_time:.2f} seconds, AVG Calculate GFLOPS: {avg_cal_flops:.2f} GFLOPS')
        print("======================== DataFlow ======================== ")
    # 释放进程组
    cleanup() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    # 是否使用预求的FLOPs
    parser.add_argument('--use-pre-calflops', type=bool, default=True)

    # 数据集所在根目录
    # "/mnt/2T/all_user/xxr/datasets/ImageNet/train_mini"
    parser.add_argument('--data-path', type=str, default="/gemini/code/Mini-ImageNet-Dataset")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/gemini/pretrain/weights/vit_base_patch16_224_in21k.pth', # './vit_base_patch16_224_in21k.pth'
                        help='initial weights path')
    # 是否冻结权重
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
