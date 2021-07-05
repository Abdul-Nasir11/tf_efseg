import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tqdm import tqdm
import cv2
from datav2 import load_data, tf_dataset
from ENet import ENet
from mask_colors import BDD_color_out
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.metrics import MeanIoU
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt





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


def class_miou(matrix):
    c1 = matrix[0,0]/(matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[2,0]) 
    c2 = matrix[1,1]/(matrix[1,1] + matrix[1,0] + matrix[1,2] + matrix[0,1] + matrix[2,1])
    c3 = matrix[2,2]/(matrix[2,2] + matrix[2,0] + matrix[2,1] + matrix[0,2] + matrix[1,2])

    return(c1,c2,c3)





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
    num_classes=3
    acc = []
    miou_total=0
    add_miou= tf.zeros( (3,3), dtype=tf.dtypes.float32, name=None)

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

        # x = x * 255.0
        # x = x.astype(np.int32)


        y=y.flatten()
        p=p.flatten()


        # accuracy = accuracy_score(y,p)
        # print(accuracy)
        # acc.append(accuracy)
    #     # print(y)
    #     # print('----'*30)
    #     # print(p)
        
        miou = MeanIoU(num_classes=3)
        miou.update_state(y, p)
        iou= np.array(miou.result())
        miou_total= miou_total+iou


        add_miou =tf.add(add_miou, miou.get_weights())
        print(f'for image: {name}, MIoU: {iou}')
        cm=confusion_matrix(y,p, num_classes=3)
        # break
        
        # miou.append(iou)
    
    # np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
    c_matrix = np.array(add_miou).reshape(num_classes,num_classes)
    c_miou= class_miou(c_matrix)
    print(c_matrix)
    print('cm')
    print(cm)
    print(c_miou)

    print('MIOU of combined : ', miou_total/len(test_x))

    df_cm = pd.DataFrame(c_matrix, range(3), range(3))
# plt.figure(figsize=(10,7))
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size

    plt.show()
    # print(acc)
    # print(len(acc))
    # total = sum(acc)/len(acc)


    
# print(len(acc))
# total_miou = sum(acc)/len(acc)


# print('Total Accuracy: ', total)
# print('Total Miou: ' total_miou)



        # output_color= BDD_color_out(p)
        # h, w, _ = x.shape
        # line = np.ones((h, 10, 3)) * 255

        # # y = np.argmax(y, axis=-1)
        # # y=y*255.0
        # gt= y.astype(np.int32)
        # # gt = np.asarray(y, dtype=np.uint8)
        # gt_color=BDD_color_out(gt)


        # # print(x.shape, line.shape, y.shape, line.shape, p.shape)
        # # pic_count=pic_count+1
        # # final_image = np.concatenate([x, line, output_color, line, gt_color], axis=1)
        # # final_image = np.concatenate([output_color, gt_color], axis=1)
        # # cv2.imwrite(f"Result/{name}.jpg", final_image)
        # # final_image.save(os.path.join(f'Result/{name}' + '_color.png'))     # original to save the prdicted image
        # # x=x.astype(np.int32)
        # # x.save(os.path.join(f'Result/{name}' + '_original.png'))     # original to save the prdicted image
        # output_color.save(os.path.join(f'Result/{name}' + '_color.png'))     # original to save the prdicted image
        # gt_color.save(os.path.join(f'Result/{name}' + '_gt.png'))     # original to save the prdicted image
