import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
from datav2 import load_data, tf_dataset
from ENet import ENet
from mask_colors import BDD_color_out

# solution on internet

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

###___________________________



H = 360
W = 640
num_classes = 3
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Dataset """
    # path = "oxford-iiit-pet/"
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    # print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    path = 'data'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")



    """ Model """
    model = ENet()
    model.load_weights('checkpoint/model_5')
    # model = tf.keras.models.load_model("model_7.h5")

    """ Saving the masks """
    pic_count=0
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split("/")[-1].split('.')[0]
        # print(name)

        ## Read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0

        ## Read mask
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (W, H))   ## (256, 256)
        y = y
        # y = np.expand_dims(y, axis=0)[0] ## (256, 256, 1)
        # y = y * (255/num_classes)
        # y = y.astype(np.int32)
        # y = np.concatenate([y, y, y], axis=2)

        ## Prediction
        p = model.predict(np.expand_dims(x, axis=0))[0]
        ## Giving colors to the masks


        p = np.argmax(p, axis=-1)
        # p = np.expand_dims(p, axis=-1)
        # p = p * (255/num_classes)
        # p = p.astype(np.int32)
        # p = np.concatenate([p, p, p], axis=2)

        x = x * 255.0
        x = x.astype(np.int32)

        output_color= BDD_color_out(p)
        h, w, _ = x.shape
        line = np.ones((h, 10, 3)) * 255

        # y = np.argmax(y, axis=-1)
        # y=y*255.0
        gt= y.astype(np.int32)
        # gt = np.asarray(y, dtype=np.uint8)
        gt_color=BDD_color_out(gt)


        # print(x.shape, line.shape, y.shape, line.shape, p.shape)
        # pic_count=pic_count+1
        # final_image = np.concatenate([x, line, output_color, line, gt_color], axis=1)
        # final_image = np.concatenate([output_color, gt_color], axis=1)
        # cv2.imwrite(f"Result/{name}.jpg", final_image)
        # final_image.save(os.path.join(f'Result/{name}' + '_color.png'))     # original to save the prdicted image
        # x=x.astype(np.int32)
        # x.save(os.path.join(f'Result/{name}' + '_original.png'))     # original to save the prdicted image
        output_color.save(os.path.join(f'Result/{name}' + '_color.png'))     # original to save the prdicted image
        gt_color.save(os.path.join(f'Result/{name}' + '_gt.png'))     # original to save the prdicted image
