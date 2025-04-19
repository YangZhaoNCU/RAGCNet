from dataloader import plydataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from utils import test_semseg
from segmentation import get_model
import random

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    """-------------------------- parameters --------------------------------------"""
    batch_size = 1
    k = 32
    """--------------------------- create Folder ----------------------------------"""
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    pred_dir = Path('./pred_global/')
    pred_dir.mkdir(exist_ok=True)
    current_time = str(datetime.datetime.now().strftime('%m-%d_%H-%M'))
    file_dir = Path(str(experiment_dir) + '/SGCNet_final')
    file_dir.mkdir(exist_ok=True)
    log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
    log_dir.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)

    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    writer = SummaryWriter(file_dir.joinpath('tensorboard'))



    torch.cuda.manual_seed(1)

    def worker_init_fn(worker_id):
        random.seed(1 + worker_id)

    """-------------------------------- Dataloader --------------------------------"""
    train_dataset = plydataset("data/train", 'train', 'SGNet')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    test_dataset = plydataset("data/test", 'test', 'SGNet')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8,worker_init_fn=worker_init_fn)

    """--------------------------- Build Network and optimizer----------------------"""
    model = get_model(cls_dim=33)
    model.cuda()
    optimizer = torch.optim.Adam(
    model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    """------------------------------------- train --------------------------------"""
    logger.info("------------------train------------------")
    best_acc = 0
    best_miou = 0
    LEARNING_RATE_CLIP = 1e-5
    his_loss = []
    his_smotth = []
    class_weights = torch.ones(15).cuda()
    for epoch in range(0, 201):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        optimizer.param_groups[0]['lr'] = lr
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            index_face, points_face, label_face, label_face_onehot, name, _, RGB_face = data
            coordinate = points_face.transpose(2, 1)
            # coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
            coordinate, label_face, index_face = Variable(coordinate.float()), Variable(label_face.long()), Variable(index_face.float())
            index_face, label_face_onehot = Variable(index_face), Variable(label_face_onehot)
            points_face = Variable(points_face.float())
            # coordinate, label_face, label_face_onehot = coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda()
            index_face, coordinate, label_face, label_face_onehot = index_face.cuda(), coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda()
            optimizer.zero_grad()

            pred,_ = model(coordinate, index_face)
            # print(pred.shape)
            # pred = model(coordinate)
            label_face = label_face.view(-1, 1)[:, 0]
            pred = pred.contiguous().view(-1, 33)

            # pred1, pred2 = model(coordinate, RGB_face)
            # _, pred = model(coordinate, RGB_face)
            # label_face = label_face.view(-1, 1)[:, 0]
            # pred = pred.contiguous().view(-1, 33)


            # label_face = label_face.view(-1, 1)[:, 0]
            # one_hot = torch.nn.functional.one_hot(label_face.unsqueeze(1)).squeeze(1).float().cuda()
            # pred1 = pred1.contiguous().view(-1, 33)
            # pred2 = pred2.contiguous().view(-1, 33)
            #
            # loss = F.nll_loss(pred1, label_face) + F.binary_cross_entropy(pred2, one_hot)

            loss = F.nll_loss(pred, label_face)
            # loss = pred
            loss.backward()
            optimizer.step()
            his_loss.append(loss.cpu().data.numpy())
        if epoch % 10 == 0:
            print('Learning rate: %f' % (lr))
            print("loss: %f" % (np.mean(his_loss)))
            writer.add_scalar("loss", np.mean(his_loss), epoch)
            torch.save(model.state_dict(), '%s/coordinate_%d.pth' % (checkpoints, epoch))
            metrics, mIoU, cat_iou, mAcc = test_semseg(model, test_loader, num_classes=33, generate_ply=True)
            print("Epoch %d, accuracy= %f, mIoU= %f, mACC= %f" % (epoch, metrics['accuracy'], mIoU, mAcc))
            logger.info("Epoch: %d, accuracy= %f, mIoU= %f, mACC= %f loss= %f" % (epoch, metrics['accuracy'], mIoU, mAcc, np.mean(his_loss)))
            writer.add_scalar("accuracy", metrics['accuracy'], epoch)
            print("best accuracy: %f best mIoU :%f, mACC: %f" % (best_acc, best_miou, mAcc))
            if ((metrics['accuracy'] > best_acc) or (mIoU > best_miou)):
                best_acc = metrics['accuracy']
                best_miou = mIoU
                print("best accuracy: %f best mIoU :%f, mACC: %f" % (best_acc, best_miou, mAcc))
                print(cat_iou)
                torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc))
                best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc)
                logger.info(cat_iou)
            his_loss.clear()
            writer.close()










