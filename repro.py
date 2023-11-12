"""
Running this script eventually gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
"""

# Net类：

# class Net(nn.Module): 定义一个名为Net的类，它继承自nn.Module，用于构建神经网络。

# def __init__(self): 定义类的初始化函数。

# super().__init__(): 调用父类nn.Module的初始化函数。

# winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5 定义一个lambda函数用于初始化权重。

# macs = 0 和 acts = 0 初始化两个变量，用于跟踪MACs（乘法累加）和激活的数量。

# self.H1w = nn.Parameter(winit(5*5*1, 12, 1, 5, 5)) 和 self.H1b = nn.Parameter(torch.zeros(12, 8, 8)) 初始化第一层的权重和偏置。

# macs += (5*5*1) * (8*8) * 12 和 acts += (8*8) * 12 更新MACs和激活的数量。

# 同样的方式初始化第二层和第三层的权重和偏置，以及更新MACs和激活的数量。

# def forward(self, x): 定义前向传播函数。

# x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) 对输入x进行填充。

# x = F.conv2d(x, self.H1w, stride=2) + self.H1b 和 x = torch.tanh(x) 对x进行卷积操作和激活函数。

# 同样的方式进行第二层和第三层的操作。

# return x 返回输出。

# if __name__ == '__main__':下的代码：

# parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits") 创建一个ArgumentParser对象。

# parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate") 和 parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs") 添加命令行参数。

# args = parser.parse_args() 解析命令行参数。

# torch.manual_seed(1337) 和 np.random.seed(1337) 设置随机数种子。

# os.makedirs(args.output_dir, exist_ok=True) 创建输出目录。

# with open(os.path.join(args.output_dir, 'args.json'), 'w') as f: 和 json.dump(vars(args), f, indent=2) 将命令行参数保存到json文件。

# model = Net() 初始化模型。

# Xtr, Ytr = torch.load('train1989.pt') 和 Xte, Yte = torch.load('test1989.pt') 加载训练和测试数据。

# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate) 初始化优化器。

# for pass_num in range(23): 进行23次训练。

# model.train() 设置模型为训练模式。

# x, y = Xtr[[step_num]], Ytr[[step_num]] 获取一个训练样本。

# yhat = model(x) 和 loss = torch.mean((y - yhat)**2) 计算预测值和损失。

# optimizer.zero_grad(set_to_none=True) 和 loss.backward() 和 optimizer.step() 计算梯度并更新参数。

# eval_split('train') 和 eval_split('test') 评估训练和测试的错误和指标。

# torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt')) 将最终的模型保存到文件。

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter # pip install tensorboardX

# -----------------------------------------------------------------------------

class Net(nn.Module):
    """ 1989 LeCun ConvNet per description in the paper """

    def __init__(self):
        super().__init__()

        # initialization as described in the paper to my best ability, but it doesn't look right...
        winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5
        macs = 0 # keep track of MACs (multiply accumulates)
        acts = 0 # keep track of number of activations

        # H1 layer parameters and their initialization
        self.H1w = nn.Parameter(winit(5*5*1, 12, 1, 5, 5))
        self.H1b = nn.Parameter(torch.zeros(12, 8, 8)) # presumably init to zero for biases
        assert self.H1w.nelement() + self.H1b.nelement() == 1068
        macs += (5*5*1) * (8*8) * 12
        acts += (8*8) * 12

        # H2 layer parameters and their initialization
        """
        H2 neurons all connect to only 8 of the 12 input planes, with an unspecified pattern
        I am going to assume the most sensible block pattern where 4 planes at a time connect
        to differently overlapping groups of 8/12 input planes. We will implement this with 3
        separate convolutions that we concatenate the results of.
        """
        self.H2w = nn.Parameter(winit(5*5*8, 12, 8, 5, 5))
        self.H2b = nn.Parameter(torch.zeros(12, 4, 4)) # presumably init to zero for biases
        assert self.H2w.nelement() + self.H2b.nelement() == 2592
        macs += (5*5*8) * (4*4) * 12
        acts += (4*4) * 12

        # H3 is a fully connected layer
        self.H3w = nn.Parameter(winit(4*4*12, 4*4*12, 30))
        self.H3b = nn.Parameter(torch.zeros(30))
        assert self.H3w.nelement() + self.H3b.nelement() == 5790
        macs += (4*4*12) * 30
        acts += 30

        # output layer is also fully connected layer
        self.outw = nn.Parameter(winit(30, 30, 10))
        self.outb = nn.Parameter(-torch.ones(10)) # 9/10 targets are -1, so makes sense to init slightly towards it
        assert self.outw.nelement() + self.outb.nelement() == 310
        macs += 30 * 10
        acts += 10

        self.macs = macs
        self.acts = acts

    def forward(self, x):

        # x has shape (1, 1, 16, 16)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = torch.tanh(x)

        # x is now shape (1, 12, 8, 8)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        slice1 = F.conv2d(x[:, 0:8], self.H2w[0:4], stride=2) # first 4 planes look at first 8 input planes
        slice2 = F.conv2d(x[:, 4:12], self.H2w[4:8], stride=2) # next 4 planes look at last 8 input planes
        slice3 = F.conv2d(torch.cat((x[:, 0:4], x[:, 8:12]), dim=1), self.H2w[8:12], stride=2) # last 4 planes are cross
        x = torch.cat((slice1, slice2, slice3), dim=1) + self.H2b
        x = torch.tanh(x)

        # x is now shape (1, 12, 4, 4)
        x = x.flatten(start_dim=1) # (1, 12*4*4)
        x = x @ self.H3w + self.H3b
        x = torch.tanh(x)

        # x is now shape (1, 30)
        x = x @ self.outw + self.outb
        x = torch.tanh(x)

         # x is finally shape (1, 10)
        return x

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))

    # init rng
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # init a model
    model = Net()
    print("model stats:")
    print("# params:      ", sum(p.numel() for p in model.parameters())) # in paper total is 9,760
    print("# MACs:        ", model.macs)
    print("# activations: ", model.acts)

    # init data
    Xtr, Ytr = torch.load('train1989.pt')
    Xte, Yte = torch.load('test1989.pt')

    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        model.eval()
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = model(X)
        loss = torch.mean((Y - Yhat)**2)
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
        writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

    # train
    for pass_num in range(23):

        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]

            # forward the model and the loss
            yhat = model(x)
            loss = torch.mean((y - yhat)**2)

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        eval_split('train')
        eval_split('test')

    # save final model to file
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
