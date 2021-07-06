import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from datav2 import load_data, tf_dataset
from ENet import ENet
# import tensorflow.keras.backend as K
# import tf.compat.v1.keras.backend as K
# from tensorflow.keras import backend as K
import keras.backend as K

continue_train = True
resume_path = 'checkpoint/model_21'
epoch_continue= 21

# solution on internet

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

###___________________________

# use CPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def weighted_categorical_crossentropy(weights):
  # weights = [0.9,0.05,0.04,0.01]
  def wcce(y_true, y_pred):
      Kweights = K.constant(weights)
      # if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
      y_true = K.cast(y_true, y_pred.dtype)
      return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
  return wcce

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Dataset """
    # path = "oxford-iiit-pet/"
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    # print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
    path = 'data/'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")


    """ Hyperparameters """
    shape = (640, 360, 3)
    num_classes = 3
    lr = 1e-4
    batch_size = 8
    epochs = 200
    class_weights = {0: 1.5201, 1:  5.0120373, 2: 7.220091}
    # cw = [ batch_size, 1.5201, 5.0120373, 7.220091]
     
    # weights = [1.5201,5.0120373, 7.220091]

    weights_nor=[1.0, 3.297176,4.749747]

    loss = weighted_categorical_crossentropy(weights_nor)

    # loss = tf.keras.losses.categorical_crossentropy(cw)
    



    if continue_train:
      cont_model=ENet()
      cont_model.load_weights(resume_path)
      cont_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr)) 
      train_dataset = tf_dataset(train_x, train_y,batch=batch_size)
      valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

      train_steps = len(train_x)//batch_size
      valid_steps = len(valid_x)//batch_size

      callbacks = [
          ModelCheckpoint("checkpoint/model_{epoch}", verbose=1, save_weights_only= True,save_best_model=True),
          ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
          # EarlyStopping(monitor="val_loss", patience=8, verbose=1),
          tf.keras.callbacks.CSVLogger('log', separator="\t", append=True)
      ]
      # class_weights={ 0: 1.52011,
      #               1: 5.0120373,
      #               2: 7.220093
      #               }
      cont_model.fit(train_dataset,

          steps_per_epoch=train_steps,
          validation_data=valid_dataset,
          validation_steps=valid_steps,
          initial_epoch=epoch_continue,
          epochs=epochs,
          callbacks=callbacks,
          # class_weight=class_weights
          # sample_weight=sample_weights
          # class_weight = 'balanced'
          
      )

   




    else:
      model = ENet()
      # model = build_unet(shape, num_classes)
      # model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), loss_weights=[1.52,5.01,7.22])
      model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr))

    # model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr))

      train_dataset = tf_dataset(train_x, train_y,batch=batch_size)
      valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

      train_steps = len(train_x)//batch_size
      valid_steps = len(valid_x)//batch_size

      callbacks = [
          ModelCheckpoint("checkpoint/model_{epoch}", verbose=1, save_weights_only= True,save_best_model=True),
          ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
          # EarlyStopping(monitor="val_loss", patience=8, verbose=1),
          tf.keras.callbacks.CSVLogger('log', separator="\t", append=True)
      ]
      # class_weights={ 0: 1.52011,
      #               1: 5.0120373,
      #               2: 7.220093
      #               }
      model.fit(train_dataset,

          steps_per_epoch=train_steps,
          validation_data=valid_dataset,
          validation_steps=valid_steps,
          epochs=epochs,
          callbacks=callbacks,
          # class_weight=class_weights
          # sample_weight=sample_weights
          # class_weight = 'balanced'
          
      )

   