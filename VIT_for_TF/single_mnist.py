import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
import sys
import math
import datetime
import time
import tensorflow as tf
from tqdm import tqdm

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 只让TensorFlow按需分配显存，而不是一次性分配所有显存
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

gflops_dict = {
    "vit_b_16_224": 1240889664,
    'vit_b_32_224': 501553920,
    "vit_l_32_224": 2104470528,
}

def main():
    data_root = "/gemini/code/mnist"  # get data root path
    # data_root = "./data/flower_photos"
    if not os.path.exists("./save_weights1"):
        os.makedirs("./save_weights1")

    batch_size = 256
    epochs = 5
    num_classes = 10
    freeze_layers = True
    initial_lr = 0.001
    weight_decay = 1e-4

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))


    total_start_time = time.time()
    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, batch_size=batch_size, val_rate=0.2)

    # create model
    model = create_model(num_classes=num_classes, has_logits=False)
    model.build((1, 224, 224, 3))

    # 下载我提前转好的预训练权重
    # 链接: https://pan.baidu.com/s/1ro-6bebc8zroYfupn-7jVQ  密码: s9d9
    # load weights
    pre_weights_path = '/gemini/pretrain/weights/ViT-B_16.h5'
    if os.path.exists(pre_weights_path):
        model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)
        # Freeze bottom layers
        if freeze_layers:
            for layer in model.layers:
                if "pre_logits" not in layer.name and "head" not in layer.name:
                    layer.trainable = False
                else:
                    print("training {}".format(layer.name))

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    # loss_object返回整个批次的平均损失。
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            # cross entropy loss
            ce_loss = loss_object(train_labels, output)

            # l2 loss，编辑正则表达式的
            matcher = re.compile(".*(bias|gamma|beta).*")
            l2loss = weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
                if not matcher.match(v.name)
            ])

            loss = ce_loss + l2loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(ce_loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    gflops, times = [], []
    eval_time_list, eval_gflops_list = [], []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Reset training metrics at the end of each epoch
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        # train_bar = tqdm(train_ds, file=sys.stdout)
        # for images, labels in train_bar:
        #     train_step(images, labels)

        #     # print train process
        #     train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
        #                                                                          epochs,
        #                                                                          train_loss.result(),
        #                                                                          train_accuracy.result())

        # update learning rate
        epoch_train_time = time.time()
        # optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        
        epoch_end_time = time.time()
        pre_cal_flops = 1240889664
        train_samples = 60000
        val_samples = 10000
        total_samples = train_samples + val_samples
        cur_epoch_time = epoch_end_time - epoch_start_time

        eval_time = epoch_end_time - epoch_train_time
        eval_gflops = pre_cal_flops * val_samples / eval_time / 1e9
        eval_gflops_list.append(eval_gflops)
        eval_time_list.append(eval_time)
        print(f"================== EVAL GFLOPS: {eval_gflops:.2f} GFLOPS, TIME: {eval_time:.2f} seconds.")

        # gflopS = pre_cal_flops * total_samples / cur_epoch_time / 1e9
        # gflops.append(gflopS)
        # times.append(cur_epoch_time)
        # print(f"EPOCH{epoch} GFLOPS: {gflopS:.2f} GFLOPS, TIME: {cur_epoch_time:.2f} seconds.")

        # writing training loss and acc
        # with train_writer.as_default():
        #     tf.summary.scalar("loss", train_loss.result(), epoch)
        #     tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # # writing validation loss and acc
        # with val_writer.as_default():
        #     tf.summary.scalar("loss", val_loss.result(), epoch)
        #     tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")


    total_end_time = time.time()
    print("Total training time: {:.2f} seconds".format(total_end_time - total_start_time))
    # print("Average Time Per Epoch: {:.2f} seconds, GFLOPS: {:.2f}".format(sum(times[-10:]) / 10 , sum(gflops[-10:]) / 10))
    cal_len = 2
    print("=================== Average Eval GFLOPS: {:.2f} GFLOPS, TIME: {:.2f} seconds".format(sum(eval_gflops_list[-cal_len:]) / cal_len , sum(eval_time_list[-cal_len:]) / cal_len))
if __name__ == '__main__':
    main()
