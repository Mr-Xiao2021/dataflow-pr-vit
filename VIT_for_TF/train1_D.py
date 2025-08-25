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
from utils1D_gpt import generate_ds

assert tf.version.VERSION >= "2.4.0", "Version of TensorFlow must be greater/equal than 2.4.0"
# 双机122.56s
# Set up the cluster configuration
tf_config = {
    "cluster": {
        "worker": ["11.11.11.15:3333", "11.11.11.10:4444"],
    },
    "task": {
        "type": "worker",
        "index": 1  # Change this index for each worker
    }
}

os.environ["TF_CONFIG"] = json.dumps(tf_config)

# Set up GPUs and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize the strategy for multi-worker training
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Enable device logging
tf.debugging.set_log_device_placement(True)

def main():
    # Load configuration from the dataset config file
    config_path = "/mnt/7T/zhaoxu/data/DriveSeg/config.json"
    with open(config_path) as config_file:
        data_config = json.load(config_file)

    num_classes = len(data_config['labels']) + 1  # Including background class

    data_root = "/mnt/7T/zhaoxu/data/DriveSeg"
    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    batch_size = 8
    epochs = 3
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
        # Create model within the strategy scope
        model = create_model(num_classes=num_classes, has_logits=False)
        model.build((1, 224, 224, 3))

        pre_weights_path = './ViT-B_16.h5'
        assert os.path.exists(pre_weights_path), f"Cannot find {pre_weights_path}"
        model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

        if freeze_layers:
            for layer in model.layers:
                if "pre_logits" not in layer.name and "head" not in layer.name:
                    layer.trainable = False
                else:
                    print(f"Training {layer.name}")

        model.summary()

        # Learning rate scheduler
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

            # 聚合所有副本的损失和准确度
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, train_loss.result(), axis=None),\
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, train_accuracy.result(), axis=None)

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

            # 聚合所有副本的损失和准确度
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, val_loss.result(), axis=None), \
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, val_accuracy.result(), axis=None)


    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # Training loop
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_loss_val, train_acc_val = train_step(images, labels)
            train_bar.desc = f"Train epoch[{epoch+1}/{epochs}] loss:{train_loss_val:.3f}, acc:{train_acc_val:.3f}"

        # Update learning rate
        optimizer.learning_rate = scheduler(epoch)


        # Validation loop
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_loss_val, val_acc_val = val_step(images, labels)
            val_bar.desc = f"Valid epoch[{epoch+1}/{epochs}] loss:{val_loss_val:.3f}, acc:{val_acc_val:.3f}"

        # Write to TensorBoard
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")

    total_end_time = time.time()
    print(f"Total training time: {total_end_time - total_start_time:.2f} seconds")

if __name__ == '__main__':
    main()
