# Author: X. Cai


from functions import (load_ped1_train_imgs_resized, load_ped1_test_imgs_resized, 
                         list2array, get_dofs, get_labels, compute_auc, compute_auc_t, compute_auc_vid)

import numpy as np
import os
import re
import cv2
from natsort import natsorted
from matplotlib import image, pylab
from PIL import Image

import keras
from keras import Input, backend as K, preprocessing
from keras.models import Model, clone_model, load_model
from keras.layers import Activation, LeakyReLU, Conv2D, Conv2DTranspose, concatenate
from keras.optimizers import Adam

from matplotlib import image, pylab, pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

from datetime import datetime

# Part 1: Data preprocessing

# Load lists of images
ped1_train_imgs = load_ped1_train_imgs_resized('../UCSDped1/Train')
ped1_test_imgs = load_ped1_test_imgs_resized('../UCSDped1/Test')
ped2_train_imgs = load_ped2_train_imgs_resized('../UCSDped2/Train')
ped2_test_imgs = load_ped2_test_imgs_resized('../UCSDped2/Test')

# Compute dense optical flow (DOF) from each two consecutive frames within each video
ped1_train_dofs = get_dofs(ped1_train_imgs)
ped1_test_dofs = get_dofs(ped1_test_imgs)
ped2_train_dofs = get_dofs(ped2_train_imgs)
ped2_test_dofs = get_dofs(ped2_test_imgs)

# Convert lists of images to arrays
ped1_train_imgs = list2array(ped1_train_imgs)
ped1_test_imgs = list2array(ped1_test_imgs)
ped2_train_imgs = list2array(ped2_train_imgs)
ped2_test_imgs = list2array(ped2_test_imgs)

# Convert lists of DOFs to arrays
ped1_train_dofs = list2array(ped1_train_dofs)
ped1_test_dofs = list2array(ped1_test_dofs)
ped2_train_dofs = list2array(ped2_train_dofs)
ped2_test_dofs = list2array(ped2_test_dofs)

# Create spatio-temporal images by concatenating each image with its corresponding DOF 
ped1_train_st_imgs = np.concatenate([ped1_train_dofs, 255-ped1_train_imgs], axis=3)
ped1_test_st_imgs = np.concatenate([ped1_test_dofs, 255-ped1_test_imgs], axis=3)
ped2_train_st_imgs = np.concatenate([ped2_train_dofs, 255-ped2_train_imgs], axis=3)
ped2_test_st_imgs = np.concatenate([ped2_test_dofs, 255-ped2_test_imgs], axis=3)

# Load ground truth labels as a list, which is needed later due to the uneven lengths of videos in Ped2
ped1_label_list = get_label('../UCSDped1/Test', 'UCSDped1.m') 
ped2_label_list = get_label('../UCSDped2/Test', 'UCSDped2.m') 

# Convert lists of labels to 1-D arrays
ped1_labels = list2array(ped1_label_list)
ped2_labels = list2array(ped2_label_list)

# Print shapes
print(ped1_train_imgs.shape, ped1_test_imgs.shape, 
      ped1_train_st_imgs.shape, ped1_test_st_imgs.shape, ped1_labels.shape)
print(ped2_train_imgs.shape, ped2_test_imgs.shape, 
      ped2_train_st_imgs.shape, ped2_test_st_imgs.shape, ped2_labels.shape)

# Normalize inputs for training and testing 
X_ped1_train_imgs = (ped1_train_imgs.astype(np.float32) - 127.5) / 127.5
X_ped1_test_imgs = (ped1_test_imgs.astype(np.float32) - 127.5) / 127.5
X_ped1_train_st_imgs = (ped1_train_st_imgs.astype(np.float32) - 127.5) / 127.5
X_ped1_test_st_imgs = (ped1_test_st_imgs.astype(np.float32) - 127.5) / 127.5

X_ped2_train_imgs = (ped2_train_imgs.astype(np.float32) - 127.5) / 127.5
X_ped2_test_imgs = (ped2_test_imgs.astype(np.float32) - 127.5) / 127.5
X_ped2_train_st_imgs = (ped2_train_st_imgs.astype(np.float32) - 127.5) / 127.5
X_ped2_test_st_imgs = (ped2_test_st_imgs.astype(np.float32) - 127.5) / 127.5

# Part 2: Build models

# Define functions for building models

def autoencoder(input_shape):
    '''Build autoencoder'''
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(64, 3, padding='same', strides=(2,2))(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, padding='same', strides=(2,2))(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 1, padding='same', strides=(2,2), name='latent_space')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(128, 3, padding='same', strides=(2,2))(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(64, 3, padding='same', strides=(2,2))(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(32, 3, padding='same', strides=(2,2))(x)
    x = LeakyReLU()(x)
    outputs = Conv2DTranspose(input_shape[-1], 3, padding='same', activation='tanh')(x)

    return Model(inputs, outputs)


def skip_autoencoder(input_shape):
    '''Build autoencoder with skip connections'''
    inputs = Input(shape=input_shape)
    x1 = Conv2D(32, 3, padding='same')(inputs)
    x2 = LeakyReLU()(x1)
    x3 = Conv2D(64, 3, padding='same', strides=(2,2))(x2)
    x4 = LeakyReLU()(x3)
    x5 = Conv2D(128, 3, padding='same', strides=(2,2))(x4)
    x6 = LeakyReLU()(x5)
    x7 = Conv2D(256, 1, padding='same', strides=(2,2), name='latent_space')(x6)
    x8 = Conv2DTranspose(128, 3, padding='same', strides=(2,2))(x7)
    x9 = LeakyReLU()(x8)
    x10 = concatenate([x6,x9])
    x11 = Conv2DTranspose(64, 3, padding='same', strides=(2,2))(x10)
    x12 = LeakyReLU()(x11)
    x13 = concatenate([x4,x12])
    x14 = Conv2DTranspose(32, 3, padding='same', strides=(2,2))(x13)
    x15 = LeakyReLU()(x14)
    x16 = concatenate([x2,x15])
    outputs = Conv2DTranspose(input_shape[-1], 3, padding='same', activation='tanh')(x16)

    return Model(inputs, outputs)


def build_model(input_shape, skip):
    '''Build ADAE when skip = 0,
    Build ADAE with skip connections in generator when skip = 1,
    Build ADAE with skip connections in both generator and discriminator when skip = 2'''
    if skip == 0:
        g_ae = autoencoder(input_shape)
        d_ae = autoencoder(input_shape)
    if skip == 1:
        g_ae = skip_autoencoder(input_shape)
        d_ae = autoencoder(input_shape)
    if skip == 2:
        g_ae = skip_autoencoder(input_shape)
        d_ae = skip_autoencoder(input_shape)
    
    # Build discriminator
    g_ae.trainable = False
    d_ae.trainable = True
    
    d_inputs_1 = Input(shape=input_shape)    
    d_inputs_2 = g_ae(d_inputs_1)
    d_outputs_1 = d_ae(d_inputs_1)
    d_outputs_2 = d_ae(d_inputs_2)

    discriminator = Model(d_inputs_1, [d_outputs_1, d_outputs_2], name="Discriminator")

    loss_d = K.mean(K.abs(d_inputs_1 - d_outputs_1)) - 0.5 * K.mean(K.abs(d_inputs_2 - d_outputs_2))
    discriminator.add_loss(loss_d)

    optimizer = Adam(lr=1e-4, beta_1=0.5)
    discriminator.compile(optimizer=optimizer)
    
    # Build gan
    g_ae.trainable = True
    d_ae.trainable = False

    gan_inputs = Input(shape=input_shape) 
    gen_outputs = g_ae(gan_inputs)
    gan_outputs_1 = d_ae(gan_inputs)
    gan_outputs_2 = d_ae(gen_outputs)

    gan = Model(gan_inputs, [gan_outputs_1, gan_outputs_2], name="GAN")

    loss_g = K.mean(K.abs(gan_inputs - gen_outputs)) + K.mean(K.abs(gen_outputs - gan_outputs_2))
    gan.add_loss(loss_g)

    gan.compile(optimizer=optimizer)
    
    return discriminator, gan

# Build 12 models for 12 experiments (3 networks x 2 types of inputs x 2 datasets)
# Each model has two sub-networks: discriminator and gan

input_shape = ped1_train_imgs.shape[1:]
model_1 = build_model(input_shape, 0)
model_2 = build_model(input_shape, 1)
model_3 = build_model(input_shape, 2)

input_shape = ped1_train_st_imgs.shape[1:]
model_4 = build_model(input_shape, 0)
model_5 = build_model(input_shape, 1)
model_6 = build_model(input_shape, 2)

input_shape = ped2_train_imgs.shape[1:]
model_7 = build_model(input_shape, 0)
model_8 = build_model(input_shape, 1)
model_9 = build_model(input_shape, 2)

input_shape = ped2_train_st_imgs.shape[1:]
model_10 = build_model(input_shape, 0)
model_11 = build_model(input_shape, 1)
model_12 = build_model(input_shape, 2)


# Part 3: Train models 

models = [model_1, model_2, model_3, model_4, model_5, model_6,
         model_7, model_8, model_9, model_10, model_11, model_12]


batch_size = 128
epochs = 100

for m, model in enumerate(models):
    # Since each model has two sub-networks, model[0] is the discriminator and model[1] is the gan
    print("Training Model_%s" % (m+1))
    
    d_loss_history = []
    g_loss_history = []
    
    for epoch in range(epochs):
        start_time = datetime.now()
        
        for i in range(X_train.shape[0]//batch_size + 1):
            if m < 3:
                images = X_ped1_train_imgs[i*batch_size : (i+1)*batch_size]
            elif m < 6:
                images = X_ped1_train_st_imgs[i*batch_size : (i+1)*batch_size]
            elif m < 9:
                images = X_ped2_train_imgs[i*batch_size : (i+1)*batch_size]
            else:
                images = X_ped2_train_st_imgs[i*batch_size : (i+1)*batch_size]
            
            # Train discriminator
            d_loss = model_1[0].train_on_batch(images, None)
            # Train gan
            g_loss = model_1[1].train_on_batch(images, None) 

        d_loss_history.append(d_loss)
        g_loss_history.append(g_loss)

        end_time = datetime.now()
        time_used = end_time - start_time
        print("epoch: %s, d_loss: %s, g_loss: %s, time_used: %s" % (epoch, d_loss, g_loss, time_used))
        
    plt.plot(d_loss_history)
    plt.plot(g_loss_history)
    plt.title('Model_%s training loss' % (m+1)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Discriminator','Generator'], loc='best')
    plt.savefig('loss_%s.png' % (m+1) )
    plt.show()

              

# Part 4: Evaluate performance 

# Compute statistics and make plots

all_fpr, all_tpr, all_fpr_t, all_tpr_t, all_auc, all_auc_t, all_auc_avg = [], [], [], [], [], [], []

for m, model in enumerate(models):
    if m < 3:
        fpr, tpr, auc_score =  compute_auc(model, X_ped1_test_imgs, m+1)
        fpr_t, tpr_t, auc_score_t = compute_auc_t(model, X_ped1_test_imgs, X_ped1_train_imgs,  m+1)
        auc_avg = compute_auc_vid(model, X_ped1_test_imgs, m+1, ped1_label_list)
        
    elif m < 6:
        fpr, tpr, auc_score =  compute_auc(model, X_ped1_test_st_imgs, m+1)
        fpr_t, tpr_t, auc_score_t = compute_auc_t(model, X_ped1_test_st_imgs, X_ped1_train_st_imgs,  m+1)
        all_auc_avg = compute_auc_vid(model, X_ped1_test_st_imgs, m+1, ped1_label_list)
    
    if m < 9:
        fpr, tpr, auc_score =  compute_auc(model, X_ped2_test_imgs, m+1)
        fpr_t, tpr_t, auc_score_t = compute_auc_t(model, X_ped2_test_imgs, X_ped2_train_imgs,  m+1)
        auc_avg = compute_auc_vid(model, X_ped2_test_imgs, m+1, ped2_label_list)
        
    else:
        fpr, tpr, auc_score =  compute_auc(model, X_ped2_test_st_imgs, m+1)
        fpr_t, tpr_t, auc_score_t = compute_auc_t(model, X_ped2_test_st_imgs, X_ped2_train_st_imgs,  m+1)
        all_auc_avg = compute_auc_vid(model, X_ped2_test_st_imgs, m+1, ped2_label_list)    
    
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(auc_score)
    all_fpr_t.append(fpr_t)
    all_tpr_t.append(tpr_t)
    all_auc_t.append(auc_score_t)
    all_auc_avg.append(auc_avg)

# Plot ROC curve

# Roc Curve on Ped1
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(6):
    plt.plot(all_fpr[i], all_tpr[i], label='Experiment %s'%(i+1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for 6 experiments on Ped1 ')
plt.legend(loc='best')
plt.savefig('ped1_roc_6.png')
plt.show()

# Roc Curve on Ped1 based on thresholded method
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(6):
    plt.plot(all_fpr_t[i], all_tpr_t[i], label='Experiment %s'%(i+1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for 6 experiments on Ped1 (threshold)')
plt.legend(loc='best')
plt.savefig('ped1_roc_t_6.png')
plt.show()

# Roc Curve on Ped2
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(6,12):
    plt.plot(all_fpr[i], all_tpr[i], label='Experiment %s'%(i+1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for 6 experiments on Ped2')
plt.legend(loc='best')
plt.savefig('ped2_roc_6.png')
plt.show()

# Roc Curve on Ped2 based on thresholded method
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(6,12):
    plt.plot(all_fpr_t[i], all_tpr_t[i], label='Experiment %s'%(i+1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for 6 experiments on Ped1 (threshold)')
plt.legend(loc='best')
plt.savefig('ped2_roc_t_6.png')
plt.show()

