import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
from ENet import ENet
from mask_colors import BDD_color_out
import time

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
   
    """ Model """
    # with tf.device("GPU:0"): 
    model = ENet()
    model.load_weights('checkpoint/model_13')
    cap = cv2.VideoCapture('Resources/park_03.mp4')


    while(cap.isOpened()):
        with tf.device("GPU:0"): 
          net, frame = cap.read()
          cv2.imshow('original video', frame)
          x = cv2.resize(frame, (W, H))
          x = x / 255.0
          x=np.expand_dims(x,axis=0)
          start = time.time()
          # p = model.predict(np.expand_dims(x, axis=0))[0]
          p = model.predict(x)[0]
          end = time.time()
          print(1/(end-start))
          p = np.argmax(p, axis=-1)
          output_color= BDD_color_out(p)
          output_color.save(os.path.join(f'image' + '_color.png'))     # original to save the prdicted image
          x = cv2.imread('image_color.png')
          cv2.imshow('Predicted', x)
          # comb_image = cv2.addWeighted(frame,1,x,1,1)
          # cv2.imshow('Combined', comb_image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    cap.release()
    cv2.destroyAllWindows()