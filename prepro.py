"""
Preprocess today's MNIST dataset into 1989 version's size/format (approximately)
http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf

Some relevant notes for this part:
- 7291 digits are used for training
- 2007 digits are used for testing
- each image is 16x16 pixels grayscale (not binary)
- images are scaled to range [-1, 1]
- paper doesn't say exactly, but reading between the lines I assume label targets to be {-1, 1}
"""

# 这段代码的主要目的是将MNIST数据集处理成1989年版本的大小和格式。MNIST是一个手写数字识别的数据集，包含60000个训练样本和10000个测试样本。

# 下面是对每一行代码的解释：

# import numpy as np：导入NumPy库，这是一个用于处理数组和矩阵的Python库。

# import torch：导入PyTorch库，这是一个用于深度学习的Python库。

# import torch.nn.functional as F：导入PyTorch的函数模块，这个模块包含了很多用于神经网络的函数，如激活函数、损失函数等。

# from torchvision import datasets：导入torchvision库的datasets模块，这个模块包含了很多常用的数据集，如MNIST、CIFAR-10等。

# torch.manual_seed(1337)和np.random.seed(1337)：设置PyTorch和NumPy的随机数种子，这样可以保证每次运行代码时，随机数的生成是一致的。

# for split in {'train', 'test'}:：对训练集和测试集进行循环处理。

# data = datasets.MNIST('./data', train=split=='train', download=True)：下载MNIST数据集，并根据当前的循环变量来选择是加载训练集还是测试集。

# n = 7291 if split == 'train' else 2007：设置训练集的大小为7291，测试集的大小为2007。

# rp = np.random.permutation(len(data))[:n]：生成一个随机排列，然后取前n个元素，这样可以随机选择n个样本。

# X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)和Y = torch.full((n, 10), -1.0, dtype=torch.float32)：创建两个全0和全-1的张量，用于存储处理后的图像和标签。

# for i, ix in enumerate(rp):：对随机选择的n个样本进行循环处理。

# I, yint = data[int(ix)]：获取当前样本的图像和标签。

# xi = torch.from_numpy(np.array(I, dtype=np.float32)) / 127.5 - 1.0：将图像转换为NumPy数组，然后转换为PyTorch张量，并将像素值缩放到[-1, 1]范围。

# xi = xi[None, None, ...]：为张量添加一个假的批次维度和通道维度。

# xi = F.interpolate(xi, (16, 16), mode='bilinear')：使用双线性插值将图像的大小调整为16x16。

# X[i] = xi[0]：将处理后的图像存储到X张量中。

# Y[i, yint] = 1.0：将正确类别的标签设置为1.0。

# torch.save((X, Y), split + '1989.pt')：将处理后的图像和标签保存为.pt文件。

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

# -----------------------------------------------------------------------------

torch.manual_seed(1337)
np.random.seed(1337)

for split in {'train', 'test'}:

    data = datasets.MNIST('./data', train=split=='train', download=True)

    n = 7291 if split == 'train' else 2007
    rp = np.random.permutation(len(data))[:n]

    X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)
    Y = torch.full((n, 10), -1.0, dtype=torch.float32)
    for i, ix in enumerate(rp):
        I, yint = data[int(ix)]
        # PIL image -> numpy -> torch tensor -> [-1, 1] fp32
        xi = torch.from_numpy(np.array(I, dtype=np.float32)) / 127.5 - 1.0
        # add a fake batch dimension and a channel dimension of 1 or F.interpolate won't be happy
        xi = xi[None, None, ...]
        # resize to (16, 16) images with bilinear interpolation
        xi = F.interpolate(xi, (16, 16), mode='bilinear')
        X[i] = xi[0] # store

        # set the correct class to have target of +1.0
        Y[i, yint] = 1.0

    torch.save((X, Y), split + '1989.pt')

