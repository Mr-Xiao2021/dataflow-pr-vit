import os
import re
import sys
import math
import datetime
import time
import tensorflow as tf
from tqdm import tqdm
"""考虑l2l loss加入优化器"""
from vit_model import vit_base_patch16_224_in21k as create_model
# from vit_model import vit_base_patch32_224_in21k as create_model
# from vit_model import vit_large_patch16_224_in21k as create_model
from utils_gpt1 import generate_ds


# export ORION_CUDART_VERSION=11.8
assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"

def main():
    strategy = tf.distribute.MirroredStrategy(["GPU:0","GPU:1","GPU:2","GPU:3","GPU:4","GPU:5","GPU:6","GPU:7"]) # [] "GPU:0","GPU:1","GPU:2","GPU:3","GPU:4","GPU:5","GPU:6","GPU:7"
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    data_root = "/gemini/code/Mini-ImageNet-Dataset"  # get data root path
    if not os.path.exists("./save_weights1"):
        os.makedirs("./save_weights1")

    batch_size = 256
    epochs = 5
    num_classes = 100
    freeze_layers = True
    initial_lr = 0.001
    weight_decay = 1e-4

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    total_start_time = time.time()

    # Adjust batch size for distributed strategy
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    # Data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, batch_size=global_batch_size, val_rate=0.2, strategy=strategy)

    with strategy.scope():
        # Create model
        model = create_model(num_classes=num_classes, has_logits=False)
        model.build((1, 224, 224, 3))

        # Load weights
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

        # Custom learning rate curve
        def scheduler(now_epoch):
            end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
            rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
            new_lr = rate * initial_lr

            with train_writer.as_default():
                tf.summary.scalar('learning rate', data=new_lr, step=epoch)

            return new_lr

        # Loss and optimizer
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        def step_fn(inputs):
            train_images, train_labels = inputs
            with tf.GradientTape() as tape:
                output = model(train_images, training=True)
                ce_loss = loss_object(train_labels, output)  # Per-replica loss
                ce_loss = tf.reduce_sum(ce_loss) / global_batch_size  # Global loss normalization

                matcher = re.compile(".*(bias|gamma|beta).*")
                l2loss = weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in model.trainable_variables
                    if not matcher.match(v.name)
                ])

                loss = ce_loss + l2loss  # Total loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)  # Log total loss
            train_accuracy.update_state(train_labels, output)  # Log accuracy

        strategy.run(step_fn, args=((train_images, train_labels),))

    @tf.function
    def val_step(val_images, val_labels):
        def step_fn(inputs):
            val_images, val_labels = inputs
            output = model(val_images, training=False)
            loss = loss_object(val_labels, output)  # Per-replica loss
            loss = tf.reduce_sum(loss) / global_batch_size  # Global loss normalization

            val_loss.update_state(loss)  # Log total loss
            val_accuracy.update_state(val_labels, output)  # Log accuracy

        strategy.run(step_fn, args=((val_images, val_labels),))


    best_val_acc = 0.
    gflops, times = [], []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # Train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # Update learning rate
        epoch_train_time = time.time()
        optimizer.learning_rate = scheduler(epoch)

        

        # Validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        epoch_end_time = time.time()
        pre_cal_flops = 1240889664
        total_samples = 38400 + 9600
        cur_epoch_time = epoch_end_time - epoch_start_time
        gflopS = pre_cal_flops * total_samples / cur_epoch_time / 1e9
        gflops.append(gflopS)
        times.append(cur_epoch_time)
        print(f"EPOCH{epoch} GFLOPS: {gflopS:.2f} GFLOPS, TIME: {cur_epoch_time:.2f} seconds.")


        # Save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights1/model.ckpt"
            model.save_weights(save_name, save_format="tf")

    total_end_time = time.time()
    print("======================== TensorFlow ======================== ")
    print("\t\tTotal E2E time: {:.2f} seconds".format(total_end_time - total_start_time))
    cal_len = 3
    print("\t\tAverage Time Per Epoch: {:.2f} seconds, GFLOPS: {:.2f}".format(sum(times[-cal_len:]) / cal_len , sum(gflops[-cal_len:]) / cal_len))
    print("======================== TensorFlow ======================== ")

if __name__ == '__main__':
    main()
