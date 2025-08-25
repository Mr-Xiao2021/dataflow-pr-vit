准备环境（已有，跳过）

conda create -n task2 python=3.7.6
conda activate task2
pip install tensorflow-gpu==2.7.0 
conda install cudatoolkit=11.2 cudnn=8.1.0

运行命令：
export ORION_CUDART_VERSION=11.8
    conda activate task2
    直接 python ***.py就行
说明：
    单卡：
        train.py和utils.py用于图像分类任务，即imagenet数据集训练。
        train1.py和utils1.py用于场景分割任务，即driveseg数据集训练。
    单机四卡：
        train_m.py图像归类
        train1_m.py场景分割
    双机八卡:
        train_D.py图像归类
        train1_D.py 场景分割




忽略下面的。
需要下载预训练权重ViT-B_16.h5：
链接: https://pan.baidu.com/s/1ro-6bebc8zroYfupn-7jVQ  密码: s9d9

实验环境：
Python = 3.7.6
TensorFlow = 2.7.0
CUDA = 11.2
cuDNN= 8.9.7
GPU = Tesla V100-SXM2-32GB×4

数据集下载：
imagenet网址：https://image-net.org/index.php
driveseg网址：https://ieee-dataport.org/open-access/mit-driveseg-manual-dataset

