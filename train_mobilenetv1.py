# encoding=UTF-8
import numpy as np
import torch
# import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import  random
from models.mobilenet import MobileNetV1
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision
'''
导入包以及设置随机种子
以类的方式定义超参数
定义自己的模型
定义早停类(此步骤可以省略)
定义自己的数据集Dataset,DataLoader
实例化模型，设置loss，优化器等
开始训练以及调整lr
绘图
预测
'''
#1. 设置随机数
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


#2. 设置超参数
class hyperParameters():
    def __init__(self):
        self.epochs = 50
        self.learning_rate = 0.001
        self.patience = 10
        self.input_size = 224
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperpara = hyperParameters()

#3. 设置模型
model = MobileNetV1(3, 10)
summary(model, input_size = (3, 224, 224), device = 'cpu')



#4. 定义早停类
class EarlyStopping():
    def __init__(self, patience = 7, verbose = False, delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss if val_loss < 0 else val_loss
        if self.best_score is None:
            self.best_score =  -val_loss
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss


# 数据集
transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#实例化模型， 设置loss, 优化器
model = model.to(hyperpara.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=hyperpara.learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

early_stopping = EarlyStopping(patience=hyperpara.patience,verbose=True)


#训练
for epoch in range(hyperpara.epochs):
    model.train()
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(trainloader, 0):
        data_x = data_x.to(torch.float32).to(hyperpara.device)
        data_y = data_y.to(torch.long).to(hyperpara.device)
        # data_y = nn.functional.one_hot(data_y.to(torch.int32), 10)
        outputs = model(data_x).softmax(1)
        optimizer.zero_grad()
        # print(data_y, outputs)
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx % 5 == 0:
            print("epoch={}/{},{}/{}of train, loss={}".format(
                epoch, hyperpara.epochs, idx, len(trainloader), loss.item()))
    train_epochs_loss.append(np.average(train_epoch_loss))

    # =====================valid============================
    model.eval()
    valid_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(testloader, 0):
        data_x = data_x.to(torch.float32).to(hyperpara.device)
        data_y = data_y.to(torch.float32).to(hyperpara.device)
        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    # ==================early stopping======================
    early_stopping(valid_epochs_loss[-1], model=model, path=r'./checkpoints/')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    # ====================adjust lr========================
    lr_adjust = {
        2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        10: 5e-7, 15: 1e-7, 20: 5e-8
    }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# =========================plot==========================
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(train_epochs_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss, '-o', label="train_loss")
plt.plot(valid_epochs_loss, '-o', label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()
# =========================save model=====================
torch.save(model.state_dict(), 'model.pth')



def pred(val):
    model = Net(1, 32, 16, 2)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    val = torch.tensor(val).reshape(1, -1).float()
    # 需要转换成相应的输入shape，而且得带上batch_size，因此转换成shape=(1,1)这样的形状
    res = model(val)
    # real: tensor([[-5.2095, -0.9326]], grad_fn=<AddmmBackward0>) 需要找到最大值所在的列数，就是标签
    res = res.max(axis=1)[1].item()
    print("predicted label is {}, {} {} 8".format(res, val.item(), ('>' if res == 1 else '<')))

