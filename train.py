
#Run commands before training on Nvidia-dgx:
#export LD_LIBRARY_PATH=/home/pandadgx/anaconda3/pkgs/cudatoolkit-11.0.221-h6bb024c_0/lib
#export LD_LIBRARY_PATH=/home/pandadgx/Downloads/cuda/lib64:$LD_LIBRARY_PATH

import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import sys

from config import *
from models.network import model_fcnn

from utils.dataio import *
from utils.process_labels import *
from utils.training_funcs import *
from utils.utils import *

#Configurations
data_config = DataConfig('config/DataConfig.json')
model_config = ModelConfig('config/ModelConfig.json')
train_config = TrainConfig('config/TrainConfig.json')
test_config = TestConfig('config/TestConfig.json')

#Load reference labels from train data
if train_config.labeltype == 'location':
    if train_config.extract_ref_labels == True:
        load_and_save_labels(path_in=data_config.source_dir_local, path_out=train_config.ref_label_path_location, labeltype=train_config.labeltype)   
    if model_config.count_num_classes:    
        NUM_CLASSES = count_classes(train_config.ref_label_path_location)
    else:
        NUM_CLASSES = model_config.num_classes_location
else:
    if train_config.extract_ref_labels == True:
        load_and_save_labels(path_in=data_config.source_dir_local, path_out=train_config.ref_label_path_class, labeltype=train_config.labeltype)   
    if model_config.count_num_classes:    
        NUM_CLASSES = count_classes(train_config.ref_label_path_class)
    else:
        NUM_CLASSES = model_config.num_classes_class

#Assign checkpointpath to save model checkpoints during training
checkpoint_path = os.path.join(train_config.checkpoint_folder, train_config.labeltype)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

#Data paths
dir_augmented = []
for augmentation in data_config.augmented_data_folders:
    dir_augmented.append(os.path.join(data_config.source_dir_augmented, augmentation, '*'+data_config.valid_extension))
dir_data = [os.path.join(data_config.source_dir_local, '*'+data_config.valid_extension)]
dir_data_test = [os.path.join(test_config.path_test_data, '*'+data_config.valid_extension)]

if train_config.train_on_gpu: #Train on gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = train_config.visible_gpus #set number of gpus
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of gpus: {}".format(mirrored_strategy.num_replicas_in_sync))

    with mirrored_strategy.scope():
        model = model_fcnn(NUM_CLASSES, 
                           input_shape=[data_config.num_freq_bin,None,3*data_config.num_audio_channel], 
                           num_filters=model_config.num_filters, 
                           wd=model_config.wd)
        model.compile(loss='categorical_crossentropy',
                      optimizer =Adam(learning_rate=train_config.max_lr, decay=train_config.decay),
                      metrics=['accuracy'])

        train_ds_obj = Dataset(dir_data + dir_augmented,
                               feat_dim=data_config.num_freq_bin,
                               hop_length=data_config.hop_length,
                               duration=data_config.duration,
                               sr=data_config.sampling_rate,
                               batch_size=train_config.batch_size,
                               num_classes=NUM_CLASSES,
                               alpha=data_config.mixup_alpha,
                               crop_percentage=data_config.crop_percentage,
                               do_mixup=train_config.do_mixup,
                               normalize=data_config.normalize,
                               shuffle=False,
                               labeltype=train_config.labeltype)
        train_ds = train_ds_obj.create_dataset()
        
        test_ds_obj = Dataset(dir_data_test, 
                              feat_dim=data_config.num_freq_bin,
                              hop_length=data_config.hop_length,
                              duration=data_config.duration,
                              sr=data_config.sampling_rate,
                              batch_size=train_config.batch_size,
                              num_classes=NUM_CLASSES,
                              alpha=data_config.mixup_alpha,
                              crop_percentage=data_config.crop_percentage,
                              do_mixup=train_config.do_mixup,
                              normalize=data_config.normalize,
                              shuffle=False,
                              labeltype=train_config.labeltype)
        test_ds = test_ds_obj.create_dataset()
else: #Train on CPU
    model = model_fcnn(NUM_CLASSES, 
                       input_shape=[data_config.num_freq_bin,None,3*data_config.num_audio_channel], 
                       num_filters=model_config.num_filters, 
                       wd=model_config.wd)

    model.compile(loss='categorical_crossentropy',
                    optimizer =Adam(learning_rate=train_config.max_lr, decay=train_config.decay),
                    metrics=['accuracy'])

    train_ds_obj = Dataset(dir_data + dir_augmented,
                            feat_dim=data_config.num_freq_bin,
                            hop_length=data_config.hop_length,
                            duration=data_config.duration,
                            sr=data_config.sampling_rate,
                            batch_size=train_config.batch_size,
                            num_classes=NUM_CLASSES,
                            alpha=data_config.mixup_alpha,
                            crop_percentage=data_config.crop_percentage,
                            do_mixup=train_config.do_mixup,
                            normalize=data_config.normalize)
    train_ds = train_ds_obj.create_dataset()

    test_ds_obj = Dataset(dir_data_test, 
                            feat_dim=data_config.num_freq_bin,
                            hop_length=data_config.hop_length,
                            duration=data_config.duration,
                            sr=data_config.sampling_rate,
                            batch_size=train_config.batch_size,
                            num_classes=NUM_CLASSES,
                            alpha=data_config.mixup_alpha,
                            crop_percentage=data_config.crop_percentage,
                            do_mixup=train_config.do_mixup,
                            normalize=data_config.normalize,
                            shuffle=False)
    test_ds = test_ds_obj.create_dataset()

#Saving checkpoints during training
save_path = os.path.join(checkpoint_path, 'cp-{epoch:02d}-{val_accuracy:.4f}.ckpt')
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, 
                                                monitor=train_config.metrics,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1,  
                                                mode='max',
                                                save_freq='epoch')

#Learning rate scheduler
if train_config.lr_scheduling_method == 'lr_warm_restart':
    lr_scheduler = LR_WarmRestart(nbatch=np.ceil(len(train_ds_obj.list_ds)/train_config.batch_size), 
                                  Tmult=2,
                                  initial_lr=train_config.max_lr, 
                                  min_lr=train_config.end_lr,
                                  epochs_restart = train_config.epochs_restart) 
    callbacks = [lr_scheduler, checkpoint]
else:
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=train_config.max_lr,
                                                                 decay_steps=train_config.decay_steps,
                                                                 end_learning_rate=train_config.end_lr,
                                                                 power=1,
                                                                 cycle=False,
                                                                 name=None)
    callbacks = [checkpoint]

history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=train_config.num_epochs, 
                    verbose=1, 
                    workers=train_config.num_workers,
                    callbacks=callbacks)

# plot history and save
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(checkpoint_path + "acc_plot.png")
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(checkpoint_path + "loss_plot.png")
plt.close()