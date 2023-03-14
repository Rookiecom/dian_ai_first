from visdom import Visdom #监听数据
import torch
import torch.nn as nn
from torch.autograd import Variable #导入自动求导机制
import torch.utils.data as Data #导入data数据集
import torchvision #导入torchvision数据集包，里面包含图像翻转等功能
import time #用于记时

#-------------------------------------------------定义超参数--------------------------------------------------------------
epoch_n = 3 #所有数据被轮到的次数
LR = 0.001 #优化算法中所用到的学习率
BatchSize = 50

#-------------------------------------------------可视化窗口设置-----------------------------------------------------------
viz = Visdom()#将窗口类实例化
#viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))#创建窗口并初始化（只监听loss）
viz.line([0.], [0], win='test_accuracy', opts=dict(title='test_accuracy'))#创建窗口并初始化（只监听accuracy）
#viz.line([[0.,0.]], [0], win='train_loss', opts=dict(title='loss&acc', legend=['loss', 'acc']))#创建窗口并初始化（loss和accuracy同时监听）

#-------------------------------------------------下载数据集torchvision.datasets.MNIST---------------------------------
train_data = torchvision.datasets.MNIST(
    root="./mnist", #root: 数据集存放位置。mnist文件夹中分为processed（包括train和test两个文件夹）和raw两个文件夹
    train=True, #True为训练集，False为测试集
    transform=torchvision.transforms.ToTensor(), #将原数据变换为(0,1)区间。并Tosensor到torch能识别的tensor类型
    download=True, #True为从网络下载数据集，若已下载则不会再次下载
)

print(train_data.train_data.size()) #输出MNIST的训练集的大小
print(train_data.train_labels.size()) #输出训练集的标签大小

#for i in range(1,4):
#    plt.imshow(train_data.train_data[i].numpy(),cmap="gray") #绘制第i张图片
#    plt.title("%i" % train_data.train_labels[i]) #添加第i张图片的标签
#    plt.show()

test_data = torchvision.datasets.MNIST( #获取测试集（下载训练集时测试集已经下载好了，这里获取一下）
    root="./mnist",
    train=False,
)

#-------------------------------------------------------加载数据集Data.DataLoader-----------------------------------------
train_loader = Data.DataLoader(dataset=train_data, #这里的Data是import torch.utils.data as Data。 dataset是已经下载好的数据集用于加载
                               batch_size=BatchSize, #加载批训练的数据个数
                               shuffle=True) #在每个epoch重新排列的数据

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1), volatile=True).type(torch.FloatTensor) #注意这里和训练集的加载方式不一样
test_y = test_data.targets #测试集的标签

#--------------------------------------------------------------定义网络结构------------------------------------------------
class CNN(nn.Module): #nn是导入的
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn) #打印创建的cnn的结构

params = list(cnn.parameters())  #打印构建的网络的参数
print(len(params))
print(params[0].size())

#----------------------------------------------------------确定优化方法和损失函数---------------------------------------------
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) #Adam优化
loss_function = nn.CrossEntropyLoss() #交叉熵损失函数

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    cnn = cnn.cuda()
time_start = time.time()

#------------------------------------------------------------进行每一轮的训练----------------------------------------------
for epoch in range(epoch_n):

    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*20) #这里的20没有具体的含义，只是用于分隔符的长度

    for batch, (x,y) in enumerate(train_loader):
        #前向传播

        if Use_gpu: #gpu可用的写法
            b_x, b_y = Variable(x.cuda()), Variable(y.cuda())
        else:
            b_x, b_y = Variable(x), Variable(y)

        #b_x, b_y = Variable(x), Variable(y)
        train_output = cnn(b_x)
        #计算训练集的损失函数和精确度
        train_loss = loss_function(train_output, b_y)

        #反向传播并优化
        optimizer.zero_grad() #每次反向传播前都要清空上一次的梯度
        train_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            if Use_gpu:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
            test_output = cnn(test_x)
            test_loss = loss_function(test_output, test_y)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            test_accuracy = sum(pred_y==test_y) / test_y.size(0)
            print("Batch:", batch, "| Train loss:%.4f"%float(train_loss), "| Test accuracy:%.4f" % test_accuracy)
            #viz.line([float(train_loss)], [batch], win='train_loss', update='append') #利用visdom监听loss
            viz.line([float(test_accuracy)], [batch], win='test_accuracy', update='append')  # 利用visdom监听accuracy
            #viz.line([[float(loss),float(accuracy)]], [step], win='train_loss', update='append') #利用visdom同时监听loss和accuracy

time_end = time.time() - time_start
print("在GPU上运行训练网络所消耗的时间(s):", time_end)
