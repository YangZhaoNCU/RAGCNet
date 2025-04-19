# *_*coding:utf-8 *_*
import os
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from pointnet2_ops import pointnet2_utils
from dataloader import generate_plyfile, plydataset

def Adj_matrix_gen(face):
    B, N = face.shape[0], face.shape[1]
    adj_1_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 0])
    adj_1_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 1])
    adj_1_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 2])
    adj_2_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 0])
    adj_2_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 1])
    adj_2_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 2])
    adj_3_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 0])
    adj_3_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 1])
    adj_3_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 2])
    adj = adj_1_1 + adj_1_2 + adj_1_3 + adj_2_1 + adj_2_2 + adj_2_3 + adj_3_1 + adj_3_2 + adj_3_3
    adj = adj.view(B, N, N)
    adj = torch.where(adj >= 1, 1., 0.)

    return adj


def furthest_point_sample(points_face, npoint):
    device = points_face.device
    xyz = points_face[:, :, 9:12]
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids #sampled pointcloud index, [B, npoint]


def fps(data, number):
    fps_idx = furthest_point_sample(data, number)
    fps_idx = fps_idx.to(torch.int)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def compute_cat_iou(pred,target,iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]  # batch_pred [N,C]
        batch_target = target[j]  # batch_target [N]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()  # index of max value  batch_choice [N]
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat, 0] += iou
            iou_tabel[cat, 1] += 1
            iou_list.append(iou)
    return iou_tabel, iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def compute_mACC(pred, label_face):
    s = 0.
    for i in range(33):
        p = torch.where(pred==i, 1., 0.)
        l = torch.where(label_face==i, 1., 0.)
        acc = (p * l).sum() / (l.sum() + 1)
        s += acc
    return s / 15


def test_semseg(model, loader, num_classes = 8, gpu=True, generate_ply=False):
    '''
    Input
    :param model:
    :param loader:
    :param num_classes:
    :param pointnet2:
    Output
    metrics: metrics['accuracy']-> overall accuracy
             metrics['iou']-> mean Iou
    hist_acc: history of accuracy
    cat_iou: IoU for o category
    '''
    iou_tabel = np.zeros((num_classes,3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    macc = 0

    shutil.rmtree('/home/zhaoyang/MAE/pred_cat')
    os.mkdir('/home/zhaoyang/MAE/pred_cat')

    for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face, RGB_face, _) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points_face = raw_points_face[0].numpy()
        index_face = index[0].numpy()
        coordinate = points.transpose(2, 1)
        normal = points[:, :, 12:]
        centre = points[:, :, 9:12]
        label_face = label_face[:, :, 0]

        index, label_face, points, coordinate = Variable(index), Variable(label_face.long()), Variable(points.float()), Variable(coordinate.float())
        index, label_face, points, coordinate = index.cuda(), label_face.cuda(), points.cuda(), coordinate.cuda()

        with torch.no_grad():
            # points = points[:, :, :12]
            # pred,_,_,_= model(points, index)
            # pred, _ = model(points, index)
            # pred,_ = model(coordinate, index)
            # pred = model(coordinate, index)
            pred = model(coordinate)

        iou_tabel, iou_list = compute_cat_iou(pred,label_face,iou_tabel)
        pred = pred.contiguous().view(-1, num_classes)
        label_face = label_face.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        macc += compute_mACC(pred_choice, label_face).cpu().data.numpy()
        correct = pred_choice.eq(label_face.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
        label_face = pred_choice.cpu().reshape(pred_choice.shape[0], 1)
        if generate_ply:

            #label_face=label_optimization(index_face, label_face)

            generate_plyfile(index_face, points_face, label_face, path=("pred_cat/%s") % name)
    iou_tabel[:,2] = iou_tabel[:,0] /(iou_tabel[:,1])
    # iou = np.where(iou_tabel<=1.)
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = ["label%d"%(i) for i in range(num_classes)]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    mIoU = np.mean(cat_iou)
    macc = macc / 58.

    return metrics, mIoU, cat_iou, macc
