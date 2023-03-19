from visdom import Visdom #监听数据
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time
import nn_replace.function as replace
epoch_n = 3 #所有数据被轮到的次数
LR = 0.001 #优化算法中所用到的学习率
BatchSize = 5

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

viz = Visdom()
viz.line([0.], [0], win='test_accuracy', opts=dict(title='test_accuracy'))#创建窗口并初始化（只监听accuracy）

train_data = torchvision.datasets.MNIST(root="./mnist", train=True, transform=torchvision.transforms.ToTensor(), download=True, )

print(train_data.train_data.size())
print(train_data.train_labels.size())



test_data = torchvision.datasets.MNIST( root="./mnist",train=False,transform=torchvision.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BatchSize, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data ,batch_size=BatchSize,shuffle=False)
'''
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1), volatile=True).type(torch.FloatTensor)
test_y = test_data.targets
test_x = test_x[:5]  # 只保留第一张图片
test_y = test_y[:5]  # 只保留第一张图片
test_x = test_x.reshape(5, 1, 28, 28) # 设置第一个维度为10，同时添加一个维度表示通道数为1
test_y = test_y.reshape(5,)
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            replace.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            replace.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out=replace.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1).to('cuda')
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

params = list(cnn.parameters())
print(len(params))
print(params[0].size())

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_function = replace.CrossEntropyLoss()

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    cnn = cnn.cuda()
time_start = time.time()

for epoch in range(epoch_n):

    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*20)
    for batch, (x,y) in enumerate(train_loader):

        if Use_gpu:
            b_x, b_y = Variable(x.cuda()), Variable(y.cuda())
        else:
            b_x, b_y = Variable(x), Variable(y)
        train_output = cnn(b_x)
        train_loss = loss_function(train_output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print("batch : ",batch,"| time:",time.time()-time_start)
        if batch % 40  == 0:
            sum=0
            correct=0
            for test_x,test_y in test_loader:
                if Use_gpu:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = cnn(test_x)
                test_loss = loss_function(test_output, test_y)
                pred_y = torch.max(test_output,1)[1].data.squeeze()
                correct+=(pred_y==test_y).sum()
                sum+=test_y.size(0)
                test_accuracy=0
                if sum==100:
                    test_accuracy=correct/sum
                    break
            #test_accuracy = sum(pred_y==test_y) / test_y.size(0)
            print("Batch:", batch, "| Train loss:%.4f"%float(train_loss), "| Test accuracy:%.4f" % test_accuracy)
            viz.line([float(test_accuracy)], [batch], win='test_accuracy', update='append')

time_end = time.time() - time_start
print("在GPU上运行训练网络所消耗的时间(s):", time_end)
