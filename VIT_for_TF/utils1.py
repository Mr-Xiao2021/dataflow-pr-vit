import os
import json
import random
import tensorflow as tf

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), f"Dataset root: {root} does not exist."

    frames_path = os.path.join(root, "frames")
    labels_path = os.path.join(root, "labels")

    assert os.path.exists(frames_path), f"Frames path: {frames_path} does not exist."
    assert os.path.exists(labels_path), f"Labels path: {labels_path} does not exist."

    frames = sorted([os.path.join(frames_path, img) for img in os.listdir(frames_path) if img.endswith(".jpg")])
    labels = sorted([os.path.join(labels_path, lbl) for lbl in os.listdir(labels_path) if lbl.endswith(".png")])

    # Debugging output
    print(f"Found {len(frames)} frames and {len(labels)} labels.")

    if len(frames) == 0 or len(labels) == 0:
        raise ValueError(f"No images or labels found in provided dataset paths: {frames_path}, {labels_path}")

    data_pairs = list(zip(frames, labels))
    random.shuffle(data_pairs)

    num_val = int(len(data_pairs) * val_rate)
    val_pairs = data_pairs[:num_val]
    train_pairs = data_pairs[num_val:]

    train_images_path, train_labels_path = zip(*train_pairs)
    val_images_path, val_labels_path = zip(*val_pairs)

    return list(train_images_path), list(train_labels_path), list(val_images_path), list(val_labels_path)

def generate_ds(data_root: str,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                val_rate: float = 0.1,
                cache_data: bool = False,
                num_classes: int = 13):
    
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    train_img_path, train_lbl_path, val_img_path, val_lbl_path = read_split_data(data_root, val_rate=val_rate)
    AUTOTUNE = tf.data.AUTOTUNE

    def process_train_info(img_path, lbl_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Changed to decode_jpeg for .jpg files
        image = tf.image.resize(image, [train_im_height, train_im_width])
        image = tf.cast(image, tf.float32) / 255.0

        label = tf.io.read_file(lbl_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, [train_im_height, train_im_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=num_classes)
        
        label = tf.reshape(label, [train_im_height, train_im_width, num_classes])
        return image, label

    def process_val_info(img_path, lbl_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Changed to decode_jpeg for .jpg files
        image = tf.image.resize(image, [val_im_height, val_im_width])
        image = tf.cast(image, tf.float32) / 255.0

        label = tf.io.read_file(lbl_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, [val_im_height, val_im_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=num_classes)
        
        label = tf.reshape(label, [val_im_height, val_im_width, num_classes])
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path, dtype=tf.string),
                                                   tf.constant(train_lbl_path, dtype=tf.string)))
    total_train = len(train_img_path)

    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_path, dtype=tf.string),
                                                 tf.constant(val_lbl_path, dtype=tf.string)))
    total_val = len(val_img_path)
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)

    def configure_for_performance(ds, shuffle_size: int, shuffle: bool = False, cache: bool = False):
        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)
        ds = ds.batch(batch_size)
        # ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)
    val_ds = configure_for_performance(val_ds, total_val, cache=cache_data)

    return train_ds, val_ds