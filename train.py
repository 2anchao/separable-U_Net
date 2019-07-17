import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datetime
import numpy as np
import matplotlib.pyplot as plt

from model_dw import U_net
from dataprocess import Mydataset
from metrics import IoUMetric
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_score = 0
torch.backends.cudnn.benchmark = True

def val(model, device, val_loader, loss, optimizer, metrics, epoch, timestamp):
    global max_score
    model.eval()
    test_loss = 0
    correct = 0
    test_miou = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            y = y.long()
            test_loss += loss(y_hat, y).item()  # sum up batch loss
            test_miou += metrics(y_hat, y)

    test_miou /= len(val_loader)
    test_loss /= len(val_loader)
    print(len(val_loader))
    writer.add_scalar('Val/Loss', test_loss, epoch)
    writer.add_scalar('Val/Miou', test_miou, epoch)

    print('\nTest set: Average loss: {:.4f}, Miou : {:.4f})\n'.format(
        test_loss, test_miou))
    if max_score < test_miou:
        max_score = test_miou
        os.makedirs('tmp/{}'.format(timestamp), exist_ok=True)
        torch.save(model, 'tmp/{}/{:.4f}_model.pth'.format(timestamp, max_score))
    return test_miou

def train(model, device, train_loader, epoch, optimizer, loss, metrics):
    total_trainloss = 0
    total_trainmiou = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        x_var = torch.autograd.Variable(x)
        #x_var=x_var.to(device)
        optimizer.zero_grad()
        y_hat = model(x_var)
        train_miou = metrics(y_hat, y.long())
        L = loss(y_hat, y.long())
        L.backward()
        optimizer.step()
        total_trainloss += float(L)
        total_trainmiou += float(train_miou)
        print("batch{}: train_miou:{:.4f} loss:{:.4f}".format(batch_idx, train_miou, L))
        if batch_idx % 10 == 0:
            niter = epoch * len(train_loder) + batch_idx
            writer.add_scalar('Train/Loss', L, niter)
            writer.add_scalar('Train/Miou', train_miou, niter)

    total_trainloss /= len(train_loder)
    total_trainmiou /= len(train_loder)
    print('Train Epoch: {}\t Loss: {:.6f}, Miou: {:.4f}'.format(epoch, total_trainloss, total_trainmiou))

if __name__ == '__main__':
    DEVICE = 'cuda'
    ACTIVATION = 'softmax'
    nb_classes = 3
    batch_size = 3
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter('log/{}'.format(timestamp))
    #数据位置
    x_train_dir = r"/home/star/ac/train_new/images"
    y_train_dir = r"/home/star/ac/train_new/masks"
    x_valid_dir = r"/home/star/ac/valid_new/images"
    y_valid_dir = r"/home/star/ac/valid_new/masks"
    # 数据读入
    train_transform = transforms.Compose([
        #transforms.Resize((192,256),2),
        transforms.ToTensor(),
        transforms.Normalize([0.519401, 0.359217, 0.310136], [0.061113, 0.048637, 0.041166]),
    ])
    valid_transform = transforms.Compose([
        #transforms.Resize((192,256),2),
        transforms.ToTensor(),
        transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526, 0.049087, 0.041330])
    ])
    train_dataset = Mydataset(images_dir=x_train_dir, masks_dir=y_train_dir, nb_classes=3, classes=[0, 1, 2],
                              transform=train_transform)
    valid_dataset = Mydataset(images_dir=x_valid_dir, masks_dir=y_valid_dir, nb_classes=3, classes=[0, 1, 2],
                              transform=valid_transform)
    train_loder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loder = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    model=U_net(3)
    criterion = nn.CrossEntropyLoss()
    metrics = IoUMetric(eps=1., activation="softmax2d")
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)
    model.cuda()
    #训练模型
    for epoch in range(0, 60):
        train(model=model, device=DEVICE, train_loader=train_loder, epoch=epoch, optimizer=optimizer, loss=criterion,
              metrics=metrics)
        test_miou = val(model=model, device=DEVICE, val_loader=valid_loder, loss=criterion, optimizer=optimizer,
                        metrics=metrics, epoch=epoch, timestamp=timestamp)
        scheduler.step(test_miou)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        print("current lr: {}".format(optimizer.param_groups[0]['lr']))
    writer.close()   

