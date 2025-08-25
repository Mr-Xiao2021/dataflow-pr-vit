import os
import json
import glob
import numpy as np
import time
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    num_classes = 250
    im_height = im_width = 224
    total_start_time = time.time()
    # load image
    # img_path = "/mnt/7T/zhaoxu/data/data1/n01440764/n01440764_18.JPEG"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # # resize image
    # img = img.resize((im_width, im_height))
    # plt.imshow(img)

    # # read image
    # img = np.array(img).astype(np.float32)

    # # preprocess
    # img = (img / 255. - 0.5) / 0.5

    # # Add the image to a batch where it's the only member.
    # img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes, has_logits=False)
    model.build([1, 224, 224, 3])

    weights_path = './save_weights1/model.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    # result = np.squeeze(model.predict(img, batch_size=1))
    # result = tf.keras.layers.Softmax()(result)
    # predict_class = np.argmax(result)

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
    #                                              result[predict_class])
    # plt.title(print_res)
    # for i in range(len(result)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               result[i]))
    # plt.show()
    data_dir = '/mnt/7T/zhaoxu/data/data1/'
    subfolders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for folder in subfolders:
        image_files = glob.glob(os.path.join(folder, '*.JPEG'))
        for img_path in image_files:
            assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
            
            # Load and preprocess image
            img = Image.open(img_path)
            img = img.resize((im_width, im_height))
            img = np.array(img).astype(np.float32)
            img = (img / 255. - 0.5) / 0.5
            img = np.expand_dims(img, 0)

            # Predict
            result = np.squeeze(model.predict(img, batch_size=1))
            result = tf.keras.layers.Softmax()(result)
            predict_class = np.argmax(result)

            # Print results
            print_res = "file: {}   class: {}   prob: {:.3}".format(
                img_path, class_indict[str(predict_class)], result[predict_class])
            print(print_res)
            # for i in range(len(result)):
            #     print("class: {:10}   prob: {:.3}".format(class_indict.get(str(i), 'Unknown'), result[i]))
    total_end_time = time.time()
    print("Total training time: {:.2f} seconds".format(total_end_time - total_start_time))

if __name__ == '__main__':
    main()
