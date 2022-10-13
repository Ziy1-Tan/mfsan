import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
num_classes = 6
iteration = 20000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/home/simple/research/datasets/"
source1_name = "1"
source2_name = '3'
target_name = "8"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
loaders = data_loader.load_training(
    root_path, batch_size, [0, 2, 7], kwargs)
source1_loader, source2_loader, target_train_loader = loaders
target_test_loader = target_train_loader


def train(model):
    # 源1、源2和目标的迭代器
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    optimizer = torch.optim.SGD(
        # 需要训练的参数
        [
            {'params': model.cls_fc_son1.parameters(), 'lr': lr[1]},
            {'params': model.cls_fc_son2.parameters(), 'lr': lr[1]},
            {'params': model.sonnet1.parameters(), 'lr': lr[1]},
            {'params': model.sonnet2.parameters(), 'lr': lr[1]},
        ],
        # 设置其他参数学习率、动量和L2权重衰减
        lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()
        optimizer.param_groups[0]['lr'] = lr[1] / \
            math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / \
            math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / \
            math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / \
            math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        # 获取batch_size大小的源1和目标样本
        try:
            source_data, source_label = next(source1_iter)
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = next(source1_iter)
        try:
            target_data, __ = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = next(target_iter)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(
            source_data).float(), Variable(source_label).long()
        target_data = Variable(target_data).float()
        # 每次batch_size清空上次的梯度
        optimizer.zero_grad()

        # 前向传播，获取分类、mmd和距离损失，计算总损失
        cls_loss, mmd_loss, l1_loss = model(
            source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        # 损失向输入侧反向传播，逐层计算梯度并保存
        loss.backward()
        # 根据梯度更新参数
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        # 计算源2的参数，同上
        try:
            source_data, source_label = next(source2_iter)
        except Exception:
            source2_iter = iter(source2_loader)
            source_data, source_label = next(source2_iter)
        try:
            target_data, _ = next(target_iter)
        except Exception:
            target_iter = iter(target_train_loader)
            target_data, _ = next(target_iter)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(
            source_data).float(), Variable(source_label).long()
        target_data = Variable(target_data).float()
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(
            source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, "to", target_name,
                  "%s max correct:" % target_name, correct.item(), "\n")


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).long()
            # 分类器cls1、cls2的预测结果
            pred1, pred2 = model(data, mark=0)

            # 按行softmax
            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            # 均值得出最终预测结果
            pred = (pred1 + pred2) / 2
            # 就是交叉熵损失
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            # 计算model在源域目标域上的正确样本并累加
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        # 计算平均loss、正确个数和正确率
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        # 源域精确度
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    return correct


if __name__ == '__main__':
    model = models.MFSAN(num_classes=num_classes)
    if cuda:
        model.cuda()
    train(model)
