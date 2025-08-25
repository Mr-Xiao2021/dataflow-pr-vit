import os
import re
import sys
import math
import datetime
import time
import json
import tensorflow as tf
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k as create_model
from utils1_gpt1 import generate_ds
assert tf.version.VERSION >= "2.4.0", "Version of TensorFlow must be greater/equal than 2.4.0"

# export ORION_CUDART_VERSION=11.8
def main():
    strategy = tf.distribute.MirroredStrategy(["GPU:0","GPU:1","GPU:2","GPU:3","GPU:4","GPU:5","GPU:6","GPU:7"])
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    num_classes = 13  # Including background class

    data_root = "/gemini/code/DriveSeg"
    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    batch_size = 8
    epochs = 5
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
    train_ds, val_ds = generate_ds(data_root, batch_size=global_batch_size, val_rate=0.2, num_classes=num_classes, strategy=strategy)

    with strategy.scope():
        # create model
        model = create_model(num_classes=num_classes, has_logits=False)
        model.build((1, 224, 224, 3))

        pre_weights_path = '/gemini/pretrain/weights/ViT-B_16.h5'
        assert os.path.exists(pre_weights_path), f"Cannot find {pre_weights_path}"
        model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

        if freeze_layers:
            for layer in model.layers:
                if "pre_logits" not in layer.name and "head" not in layer.name:
                    layer.trainable = False
                else:
                    print(f"training {layer.name}")

        # model.summary()

        def scheduler(now_epoch):
            end_lr_rate = 0.01  
            rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate 
            new_lr = rate * initial_lr

            with train_writer.as_default():
                tf.summary.scalar('learning rate', data=new_lr, step=epoch)

            return new_lr

        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        def step_fn(inputs):
            train_images, train_labels = inputs
            with tf.GradientTape() as tape:
                output = model(train_images, training=True)
            
                output_shape = tf.shape(output)
                batch_size = output_shape[0]
                num_classes = output_shape[1]

                train_labels_flat = tf.reshape(train_labels, [batch_size, -1, num_classes])
                train_labels_flat = tf.reduce_mean(train_labels_flat, axis=1)

                ce_loss = loss_object(train_labels_flat, output)
                matcher = re.compile(r".*(bias|gamma|beta).*")
                l2loss = weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in model.trainable_variables
                    if not matcher.match(v.name)
                ])
                loss = tf.reduce_sum(ce_loss) / global_batch_size + l2loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(ce_loss)
            train_accuracy.update_state(train_labels_flat, output)

        strategy.run(step_fn, args=((train_images, train_labels),))

    @tf.function
    def val_step(val_images, val_labels):
        def step_fn(inputs):
            val_images, val_labels = inputs
            output = model(val_images, training=False)

            output_shape = tf.shape(output)
            batch_size = output_shape[0]
            num_classes = output_shape[1]

            val_labels_flat = tf.reshape(val_labels, [batch_size, -1, num_classes])
            val_labels_flat = tf.reduce_mean(val_labels_flat, axis=1)

            loss = loss_object(val_labels_flat, output)

            val_loss.update_state(loss)
            val_accuracy.update_state(val_labels_flat, output)

        strategy.run(step_fn, args=((val_images, val_labels),))

    best_val_acc = 0.
    gflops, times = [], []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}] loss:{train_loss.result():.3f}, acc:{train_accuracy.result():.3f}"

        optimizer.learning_rate = scheduler(epoch)

        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)
            val_bar.desc = f"valid epoch[{epoch+1}/{epochs}] loss:{val_loss.result():.3f}, acc:{val_accuracy.result():.3f}"


        epoch_end_time = time.time()
        pre_cal_flops = 1240889664
        total_samples = 5000
        
        cur_epoch_time = epoch_end_time - epoch_start_time
        gflopS = pre_cal_flops * total_samples / cur_epoch_time / 1e9
        gflops.append(gflopS)
        times.append(cur_epoch_time)
        print(f"EPOCH{epoch+1} GFLOPS: {gflopS:.2f} GFLOPS, TIME: {cur_epoch_time:.2f} seconds.")


        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")

    total_end_time = time.time()
    print("======================== TensorFlow ======================== ")
    print("\t\tTotal E2E time: {:.2f} seconds".format(total_end_time - total_start_time))
    cal_len = 5
    print("\t\tAverage Time Per Epoch: {:.2f} seconds, GFLOPS: {:.2f}".format(sum(times[-cal_len:]) / cal_len , sum(gflops[-cal_len:]) / cal_len))
    print("======================== TensorFlow ======================== ")

if __name__ == '__main__':
    main()
