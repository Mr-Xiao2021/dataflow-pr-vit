import os
import json
import random

import tensorflow as tf


def read_split_data(root: str):
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")

    # 遍历 train 文件夹获取类别名
    flower_class = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]
    flower_class.sort()
    class_indices = {k: v for v, k in enumerate(flower_class)}

    # 保存索引映射为 JSON
    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    supported = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".bmp"]

    def get_images_and_labels(folder, class_indices):
        images_path = []
        images_label = []
        for cla in class_indices.keys():
            cla_path = os.path.join(folder, cla)
            if not os.path.exists(cla_path):
                continue
            images = [os.path.join(cla_path, img) for img in os.listdir(cla_path)
                      if os.path.splitext(img)[-1] in supported]
            label = class_indices[cla]
            images_path.extend(images)
            images_label.extend([label] * len(images))
        return images_path, images_label

    train_images_path, train_images_label = get_images_and_labels(train_root, class_indices)
    val_images_path, val_images_label = get_images_and_labels(val_root, class_indices)

    print(f"Found {len(train_images_path)} training images, {len(val_images_path)} validation images.")

    return train_images_path, train_images_label, val_images_path, val_images_label


def generate_ds(data_root: str,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                val_rate: float = 0.1,
                cache_data: bool = False,
                strategy=None):
    """
    读取划分数据集，并生成训练集和验证集的迭代器
    :param data_root: 数据根目录
    :param train_im_height: 训练输入网络图像的高度
    :param train_im_width:  训练输入网络图像的宽度
    :param val_im_height: 验证输入网络图像的高度
    :param val_im_width:  验证输入网络图像的宽度
    :param batch_size: 训练使用的batch size
    :param val_rate:  将数据按给定比例划分到验证集
    :param cache_data: 是否缓存数据
    :param strategy: 分布式策略对象 (如 tf.distribute.MirroredStrategy)
    :return:
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    train_img_path, train_img_label, val_img_path, val_img_label = read_split_data(data_root)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def process_train_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, train_im_height, train_im_width)
        image = tf.image.random_flip_left_right(image)
        image = (image / 255. - 0.5) / 0.5
        return image, label

    def process_val_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, val_im_height, val_im_width)
        image = (image / 255. - 0.5) / 0.5
        return image, label

    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path),
                                                   tf.constant(train_img_label)))
    total_train = len(train_img_path)

    # Use Dataset.map to create a dataset of image, label pairs
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_path),
                                                 tf.constant(val_img_label)))
    total_val = len(val_img_path)
    # Use Dataset.map to create a dataset of image, label pairs
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)

    if strategy is not None:
        # 将数据集分发到多卡
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)

    return train_ds, val_ds
