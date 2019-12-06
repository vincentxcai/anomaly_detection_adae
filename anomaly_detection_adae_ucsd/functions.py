# Author: X.Cai

import numpy as np
import os
import re
import cv2
from natsort import natsorted
from matplotlib import image, pylab
from PIL import Image

# Part 1: Data preprocessing functions

def load_ped1_train_imgs(path):
    '''load train images into a list of lists'''
    
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Train'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = image.imread(path + '/' + folder + '/' + filename)
                    img = img[6:,:-6]
                    img = img.reshape(img.shape + (1,))
                    video.append(img) 
            all_videos.append(video)
    return all_videos

def load_ped1_test_imgs(path):
    '''load train images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Test') and not folder.endswith('gt'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = image.imread(path + '/' + folder + '/' + filename)
                    img = img[6:,:-6]
                    img = img.reshape(img.shape + (1,))
                    video.append(img)
            all_videos.append(video)
    return all_videos


def load_ped2_train_imgs(path):
    '''load train images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Train'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = image.imread(path + '/' + folder + '/' + filename)
                    img = img.reshape(img.shape + (1,))
                    video.append(img) 
            all_videos.append(video)
    return all_videos

def load_ped2_test_imgs(path):
    '''load train images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Test') and not folder.endswith('gt'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = image.imread(path + '/' + folder + '/' + filename)
                    img = img.reshape(img.shape + (1,))
                    video.append(img)
            all_videos.append(video)
    return all_videos

def load_ped1_train_imgs_resized(path):
    '''load train images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Train'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = Image.open(path + '/' + folder + '/' + filename)
                    img.load()
                    img.thumbnail([152, 232])
                    img = np.asarray(img, dtype="int32" )
                    img = img[:96,:]
                    img = img.reshape(img.shape + (1,))
                    video.append(img) 
            all_videos.append(video)
    return all_videos

def load_ped1_test_imgs_resized(path):
    '''load test images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Test') and not folder.endswith('gt'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = Image.open(path + '/' + folder + '/' + filename)
                    img.load()
                    img.thumbnail([152, 232])
                    img = np.asarray(img, dtype="int32" )
                    img = img[:96,:]
                    img = img.reshape(img.shape + (1,))
                    video.append(img)
            all_videos.append(video)
    return all_videos


def load_ped2_train_imgs_resized(path):
    '''load train images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Train'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = Image.open(path + '/' + folder + '/' + filename)
                    img.load()
                    img.thumbnail([152, 96])
                    img = np.asarray(img, dtype="int32" )
                    img = img.reshape(img.shape + (1,))
                    video.append(img) 
            all_videos.append(video)
    return all_videos

def load_ped2_test_imgs_resized(path):
    '''load test images into a list of lists'''
    all_videos = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Test') and not folder.endswith('gt'): 
            video = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    img = Image.open(path + '/' + folder + '/' + filename)
                    img.load()
                    img.thumbnail([152, 96])
                    img = np.asarray(img, dtype="int32" )
                    img = img.reshape(img.shape + (1,))
                    video.append(img)
            all_videos.append(video)
    return all_videos

def list2array(img_list):
    '''turn a list of lists into a numpy array'''
    img_arr = []
    for seq in img_list:
        img_arr.extend(seq)
    return np.array(img_arr)

def get_dof(im1, im2):
    '''compute a dense optical flow from 2 images'''
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def get_dofs(imgs):
    '''compute all dense optical flows from a list of images'''
    all_dofs = []
    for seq in imgs:
        dofs = []
        for i in range(len(seq)):
            if i < len(seq)-1:
                dof = get_dof(seq[i], seq[i+1])
            else:
                dof = get_dof(seq[i], seq[i])
            dofs.append(dof)
        all_dofs.append(dofs)
    return all_dofs


def get_labels(path, file):
    '''Create labels from the matlab file in the test set'''
    with open(path + file) as file:
        lines = file.readlines() 
    all_labels = []
    for folder in natsorted(os.listdir(path)):
        if folder.startswith('Test') and not folder.endswith('gt'):
            labels = []
            for filename in natsorted(os.listdir(path + '/' + folder)):
                if filename.endswith('.tif'): 
                    labels.append(0)
            all_labels.append(labels)
    for i, line in enumerate(lines[1:]):
        indices = re.findall(r'(\d+):(\d+)',line)[0]
        for j in range(int(indices[0])-1, int(indices[1])):
            all_labels[i][j] = 1
    return all_labels

# Part 2: Evaluation functions

def compute_auc(model, X_test, exp_num):
    '''Calculate AUC over all frames and make plots'''
    
    pred_imgs = model[1].predict(X_test)[1] 
    recon_errors = []
    for i in range(X_test.shape[0]):
        recon_errors.append(np.sqrt(np.sum(np.square(pred_imgs[i] - X_test[i]))))

    recon_errors = np.array(recon_errors)
    anomaly_scores = (recon_errors - min(recon_errors)) / (max(recon_errors) - min(recon_errors))

    fpr, tpr, _ = roc_curve(gt_labels, anomaly_scores)
    auc_score = auc(fpr, tpr)

    print("AUC: %.2f%%" % (100* auc_score))

    plt.figure(figsize=(14,3))
    plt.plot(anomaly_scores, label='Anomaly Score')
    plt.plot(gt_labels, label='Ground Truth')
    plt.legend()
    plt.title('Experiment %s:    AUC: %.2f%% '%(exp_num, (100* auc_score))) 
    plt.savefig('auc_%s.png'% exp_num) 
    plt.show()
    
    return fpr, tpr, auc_score

def compute_auc_t(model, X_test, X_train, exp_num):
    '''Calculate AUC over all frames but only consider pixel error over a threshold, and make plots'''
    pred_imgs = model[1].predict(X_test)[1] 
    preds_train = model[1].predict(X_train)[1] 

    # Compute a threshold, pixels with error higher than this threshold is more likely to be anomalous
    thres_mean = np.mean(np.abs(preds_train - X_train))
    thres_std = np.std(np.abs(preds_train - X_train))
    threshold = thres_mean + thres_std

    anomaly_scores_t = []
    for i in range(X_test.shape[0]):
        recon_errors_abs = np.abs(pred_imgs[i] - X_test[i])
        recon_errors_t = recon_errors_abs[recon_errors_abs > threshold]
        anomaly_scores_t.append(np.mean(recon_errors_t))

    anomaly_scores_t = (anomaly_scores_t - min(anomaly_scores_t)) / (max(anomaly_scores_t) - min(anomaly_scores_t))

    fpr_t, tpr_t, _ = roc_curve(gt_labels, anomaly_scores_t)
    auc_score_t = auc(fpr_t, tpr_t)

    print("AUC_t: %.2f%%" % (100* auc_score_t))

    plt.figure(figsize=(14,3))
    plt.plot(anomaly_scores, label='Anomaly Score')
    plt.plot(gt_labels, label='Ground Truth')
    plt.legend()
    plt.title('Experiment %s:    AUC_t: %.2f%% '%(exp_num, (100* auc_score_t))) 
    plt.savefig(path_experiment + 'auc_t_%s.png'% exp_num) 
    plt.show()
    
    return fpr_t, tpr_t, auc_score_t


def compute_auc_vid(model, X_test, exp_num, label_list):
    '''Calculate AUC for each video and make plots'''
    pred_imgs = model[1].predict(X_test)[1] 
    recon_errors = []
    for i in range(X_test.shape[0]):
        recon_errors.append(np.sqrt(np.sum(np.square(pred_imgs[i] - X_test[i]))))
    recon_errors = np.array(recon_errors)
    all_video_aucs = []
    start = 0
    for i, video_labels in enumerate(label_list):
        end = start + len(video_labels)               
        video_scores = recon_errors[start:end]
        video_scores = (video_scores - min(video_scores))/(max(video_scores) - min(video_scores))
        start += len(video_labels)
        if 0 not in video_labels:
            video_labels[0] = 0
        fpr_v, tpr_v, _ = roc_curve(video_labels, video_scores)
        auc_score_v = auc(fpr_v, tpr_v)
        print('Video %s   AUC: %.2f%%'%((i+1), 100*auc_score_v))
        all_video_aucs.append(auc_score_v)
        
        plt.plot(video_scores, label='Anomaly Score')
        plt.plot(video_labels, label='Ground Truth')
        plt.legend()
        plt.title('Experiment %s  Video %s   AUC: %.2f%%'%(exp_num, (i+1), 100*auc_score_v)) 
        plt.savefig(video_path + 'exp_%s_vid_%s_auc.png'%(exp_num, (i+1))) #!!!
        plt.show()

    auc_avg = np.mean(np.array(all_video_aucs))
    print('Average AUC: %.2f%%'% (100*auc_avg))
    
    return auc_avg
