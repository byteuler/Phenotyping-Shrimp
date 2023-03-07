import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from GCNv import GCN
import cfg0 as cfg
from metrics import *
from matplotlib import pyplot as plt
import os.path as osp
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)


gcn = GCN(num_class)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(gcn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5,patience=10,cooldown=0,min_lr=0, eps=1e-08)

def train(model):

    model.to(device)
    net = model.train()
    best = [0]
    val_interval = 1
    train_curve = list()
    valid_curve = list()
    train_epoch_curve = list()
    train_miou = list()
    val_miou = list()
    running_metrics_train = runningScore(n_classes = num_class)

    for epoch in range(cfg.EPOCH_NUMBER):
        print(datetime.now())
        print(best)
        running_metrics_train.reset()
        train_loss = 0.
        train_epoch_loss =0.

        for i, sample in enumerate(train_data):

            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_curve.append(loss.item())
            train_epoch_loss += (loss.item()*len(sample['img']))

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = img_label.data.cpu().numpy()
            running_metrics_train.update(true_label, pre_label)


        train_epoch_loss/=len(Load_train)

        train_epoch_curve.append(train_epoch_loss)
        metrics = running_metrics_train.get_scores()
        train_miou.append(metrics[0]['mIou: '])

        print('Training: Epoch[{:0>3}/{:0>3}] Loss: {:.4f}\t'.format(epoch + 1, cfg.EPOCH_NUMBER, train_epoch_loss))

        for k, v in metrics[0].items():
            print('Training: Epoch[{:0>3}/{:0>3}]\t'.format(epoch + 1, cfg.EPOCH_NUMBER), k, v)
        print('Training: Epoch[{:0>3}/{:0>3}]\t'.format(epoch + 1, cfg.EPOCH_NUMBER), metrics[1])
        print('Training: Epoch[{:0>3}/{:0>3}]\tLR: {}'.format(epoch + 1, cfg.EPOCH_NUMBER,optimizer.param_groups[0]['lr']))

        if (epoch +1 ) % val_interval == 0:
            val_iou = evaluate(model,best,epoch,valid_curve)
            val_miou.append(val_iou)
        scheduler.step(val_iou)

def evaluate(model,best,epoch,valid_curve):
    """
    :param model:
    :param best:
    :param epoch:
    :param valid_curve:
    :return:
    """
    print(datetime.now())
    net = model.eval()
    running_metrics_val = runningScore(n_classes =num_class)
    eval_loss = 0
    prec_time = datetime.now()
    with torch.no_grad():
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].to(device))
            valLabel = Variable(sample['label'].long().to(device))

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = valLabel.data.cpu().numpy()
            running_metrics_val.update(true_label, pre_label)
        loss_val_mean = eval_loss / len(val_data)
        valid_curve.append(loss_val_mean)
        metrics = running_metrics_val.get_scores()
    cur_time = datetime.now()
    print("Valid: Epoch[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
        epoch+1, cfg.EPOCH_NUMBER,  loss_val_mean ))


    for k, v in metrics[0].items():
        print('Valid: ',k, v)
    print(metrics[1])

    val_miou = metrics[0]['mIou: ']
    if max(best) <= val_miou:
        best.append(val_miou)
        torch.save(net.state_dict(), osp.join(cfg.Weight_Path,'{}.pth'.format(epoch+1)))


    h, remainder = divmod((cur_time - prec_time).total_seconds(), 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Runing Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print('Valid: ',time_str)
    return val_miou



if __name__ == "__main__":

    train(gcn)



