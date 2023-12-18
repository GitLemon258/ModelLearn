import torch
import torchvision
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
# import queue

#设置随机种子
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#添加tensorboard
writer = SummaryWriter("logs_MoCo")

#数据增强设置：从随机调整大小的图像中取128*128像素的作物，然后进行随机颜色抖动、随机水平翻转和随机灰度转换
transform_aug = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        #以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
        transforms.RandomHorizontalFlip(),
        #将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
        transforms.RandomResizedCrop(size=96),
        transforms.Resize((224, 224)),
        #随机应用给定概率的变换列表  随机改变一个图像的亮度、对比度、饱和度和色调
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        #以一定的概率将图像变为灰度图像
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

#获取数据
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)
train_dataset = torchvision.datasets.CIFAR10(root='../datasetCIFAR10',train=True,transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../datasetCIFAR10',train=False,transform=transform,download=True)
#加载数据集
batch_size = 2
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)


Resnet18 = torchvision.models.resnet18(weights=None)
Resnet18.fc = nn.Linear(Resnet18.fc.in_features,224)
model = nn.Sequential(
    Resnet18,
    nn.Linear(224,224*4),
    nn.ReLU(),
    nn.Linear(224*4,224)
)

#初始化encoder
#克隆encoder！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
encoder_q = model.cuda()
# encoder_k = model.cuda()
# # encoder_k = encoder_q.clone()
# for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
#     param_k.data.copy_(param_q.data)
#     param_k.requires_grad = False
encoder_k = copy.deepcopy(encoder_q)

def _momentum_update_encoder_k(encoder_q,encoder_k,m=0.999):
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

'''
K = 65536
queue_dic = torch.randn(128, K)
queue_dic = nn.functional.normalize(queue_dic, dim=0)
# 队列的指针
ptr = 0
def Queue_update(keys):
    batch_size = keys.shape[0]
    #assert:与其让它在运行时崩溃，不如在出现错误条件时就崩溃
    assert K % batch_size == 0
    #global关键字来声明全局变量(def内部需要修改全局变量）
    global ptr
    queue_dic[:, ptr : ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % K
'''

def InfoNCE(query,keys):
    with torch.no_grad():
        # print("query.shape  ",query.shape)
        # print("key.shape  ",keys.shape)
        # positive logits: Nx1
        # tensor.unsqueeze(dim=a)用途：进行维度扩充，在指定位置加上维数为1的维度 参数设置：如果设置dim=a，就是在维度为a的位置进行扩充 如果dim为负，则将会被转化dim+input.dim()+1
        l_pos = torch.einsum("nc,nc->n", [query, keys]).unsqueeze(-1)
        # negative logits: NxK
        # l_neg = torch.einsum("nc,ck->nk", [query, keys])
        keys = keys.data.cpu().numpy()
        # print("l_pos.shape  ",l_pos.size())
        l_neg = torch.zeros((batch_size, 224)).cuda()
        for i in range(batch_size):
            keys_neg = torch.from_numpy(np.delete(keys,i,0)).cuda()
            # print("query[i].shape  ",query[i].size())
            # print("keys_neg.shape  ",keys_neg.size())
            # print(query[i])
            # print(keys_neg)
            l_neg[i] = torch.einsum("n,bn->n", [query[i], keys_neg])/(batch_size-1)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg],dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels

#The SGD weight decay is 0.0001 and the SGD momentum is 0.9. For IN-1M, we use a mini-batch size of 256 (N in Algorithm 1) in 8 GPUs, and an initial learning rate of 0.03. We train for 200 epochs
epochs = 200
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
temperature = 0.07
x_dic = torch.zeros((1, 3, 224,224)).cuda()
flag_DeleFirstLine = True
for epoch in range(epochs):
    model.train()
    x_dic = torch.zero_(x_dic)
    for img, label in train_loader:
        x_q = torch.zeros((batch_size, 3, 224, 224)).cuda()
        x_k = torch.zeros((batch_size, 3, 224, 224)).cuda()
        for i in range(img.shape[0]):
            #transforms.ToPILImage()返回值是一个类，(img[i])是调用该类中的函数，img[i]是参数
            x_q[i] = transform_aug(transforms.ToPILImage()(img[i]))
            x_k[i] = transform_aug(transforms.ToPILImage()(img[i]))
        x_dic = torch.cat((x_dic,x_k)).cuda()
        #删除x_dic的第一行！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # x_dic = x_dic[1:].cuda()
        if flag_DeleFirstLine:
            x_dic = torch.narrow(x_dic,0,1,batch_size).cuda()
            flag_DeleFirstLine = False
        if torch.cuda.is_available():
            x_q = x_q.cuda()
            x_k = x_k.cuda()
        output_q = torch.nn.functional.normalize(encoder_q(x_q)).cuda()
        output_k = torch.nn.functional.normalize(encoder_k(x_dic)).cuda()
        logits, label = InfoNCE(output_q, output_k)
        logits = logits.cuda()
        label = label.cuda()
        loss = torch.nn.CrossEntropyLoss()(logits/temperature, label).cuda()
        # 梯度归零
        optimizer.zero_grad()
        # 计算每个参数的梯度??????????????????????????????????
        # loss.backward()
        # 执行一次优化步骤
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            _momentum_update_encoder_k(encoder_q,encoder_k)
        # Queue_update(output_k)
        writer.add_scalar('loss', loss, epoch)
        print("loss",loss)
        print("运行")

writer.close()