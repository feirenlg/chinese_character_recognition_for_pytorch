import torch
import os
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import DataSet
from model.network import Network
from utils.visualizer import Visualizer
from torchnet import meter
from tqdm import tqdm
from config import conf
from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
        transforms.Resize((conf.RE_SIZE, conf.RE_SIZE)),
        transforms.Pad(conf.PADDING_SIZE, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[conf.MEAN_R, conf.MEAN_G, conf.MEAN_B],
                             std=[conf.STD_R, conf.STD_G, conf.STD_B])
    ])

def train(**kwargs):
    conf.parse(kwargs)

    # train_set = DataSet(cfg, train=True, test=False)
    train_set = ImageFolder(conf.TRAIN_DATA_ROOT, transform)
    train_loader = DataLoader(train_set, conf.BATCH_SIZE,
                              shuffle=True,
                              num_workers=conf.NUM_WORKERS)

    model = Network()

    if conf.LOAD_MODEL_PATH:
        print(conf.LOAD_MODEL_PATH)
        model.load_state_dict(torch.load(conf.CHECKPOINTS_ROOT + conf.LOAD_MODEL_PATH))

    device = torch.device('cuda:0' if conf.USE_GPU else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    lr = conf.LEARNING_RATE
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=lr,
                             weight_decay=conf.WEIGHT_DECAY)
    model.to(device)

    for epoch in range(conf.MAX_EPOCH):

        model.train()
        running_loss = 0
        for step, (inputs, targets) in tqdm(enumerate(train_loader)):

            inputs, targets = inputs.to(device), targets.to(device)
            optim.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, targets)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if step % conf.PRINT_FREQ == conf.PRINT_FREQ - 1:
                running_loss = running_loss / conf.PRINT_FREQ
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss))
                # vis.plot('loss', running_loss)
                running_loss = 0



        torch.save(model.state_dict(), conf.CHECKPOINTS_ROOT + time.strftime('%Y-%m-%d-%H-%M-%S.pth'))

        for param_group in optim.param_groups:
            lr *= conf.LEARNING_RATE_DECAY
            param_group['lr'] = lr


def val(model, loader):

    confusion_matrix = meter.ConfusionMeter(conf.MAX_CLASSES)
    model.eval()
    with torch.no_grad():
        for step, (data, target) in tqdm(enumerate(loader)):
            device = torch.device('cuda:0' if conf.USE_GPU else 'cpu')
            val_input = data.to(device)
            out = model(val_input)
            confusion_matrix.add(out.detach(), target.detach())
    model.train()
    cm_value = confusion_matrix.value()
    correct_sum = 0
    for i in range(conf.MAX_CLASSES):
        correct_sum += cm_value[i][i]
    accuracy = 100. * correct_sum / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    conf.parse(kwargs)

    model = Network().eval()

    if conf.LOAD_MODEL_PATH:
        print(conf.LOAD_MODEL_PATH)
        model.load_state_dict(torch.load(conf.CHECKPOINTS_ROOT + conf.LOAD_MODEL_PATH))

    device = torch.device('cuda:0' if conf.USE_GPU else 'cpu')
    model.to(device)

    test_set = ImageFolder(conf.TEST_DATA_ROOT, transform)
    test_loader = DataLoader(test_set, conf.BATCH_SIZE,
                             shuffle=False,
                             num_workers=conf.NUM_WORKERS)

    results = list()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = model(inputs)
            pred = torch.max(outs, 1)[1]
            # print((targets == pred).float())
            # (prob_top_k, idxs_top_k) = probability.topk(3, dim=1)

            acc = (pred == targets).float().sum() / len(targets)
            results += ((pred == targets).float().to('cpu').numpy().tolist())

            print('[%5d] acc: %.3f' % (step + 1, acc))

        results = np.array(results)
        print('Top 1 acc: %.3f' % (np.sum(results) / len(results)))


if __name__ == '__main__':
    import fire

    fire.Fire()
