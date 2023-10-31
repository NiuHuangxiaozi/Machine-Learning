import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.datasets as da
import torchvision
import torch.utils.data as Data
from tqdm import *
import os
import torch.nn.functional as F
import onnx


class FlowerDataset:
    def __init__(self, dir_path, train_batch=32, test_batch=15):
        self.dir_path = dir_path
        self.train_batch_size = train_batch
        self.test_batch_size = test_batch
        self.classification = {
            "daisy": 0,
            "roses": 1,
            "sunflowers": 2,
            "tulips": 3
        }

        self.train_size = None
        self.test_size = None

    def process_data(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
        train_set = da.ImageFolder(self.dir_path + "/flower_photos", transform=augs)
        test_set = da.ImageFolder(self.dir_path + "/flower_test_photos", transform=augs)

        self.train_size = len(train_set)
        self.test_size = len(test_set)
        train_iter = Data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)
        test_iter = Data.DataLoader(test_set, batch_size=self.test_batch_size, shuffle=True)

        return train_iter, test_iter

    def Get_data_size(self):
        return self.train_size, self.test_size


class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, padding=1, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, padding=1, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, padding=1, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.linear = nn.Linear(in_features=12 * 28 * 28, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x




if __name__ == "__main__":
    # 设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataset = FlowerDataset(os.getcwd())
    train_loader, test_loader = dataset.process_data()
    train_length, test_length = dataset.Get_data_size()
    # 定义模型
    model = Simple_CNN()
    model = model.to(device)

    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 定义优化器，梯度更新规则
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # 我们一共训练10轮
    Epoch = 10

    # history记录test阶段的loss和accurary用于画图
    history = {'Test Loss': [], 'Test Accuracy': []}

    for epoch in range(1, Epoch + 1):
        # 训练部分
        '''
            这个写法可以生成进度条，enumerrate会生成（index，value)的组合对
            trainloader对象本来就是（data，label）的组合对
        '''
        processBar = tqdm(enumerate(train_loader), total=len(train_loader))

        # 模型训练之前一定要写
        model.train()

        for index, (data, label) in processBar:
            '''
               data的形状: [B,3,224,224]
            '''
            data = data.to(device)
            label = label.to(device)

            # 模型前向传播
            outputs = model(data)

            # argmax就是按照某一个维度求得这个维度上最大值的下标，如果不想降维，请使用keepdim=True
            prediction = torch.argmax(outputs, dim=1)

            # sum(prediction==label)会生成0-1矩阵，sum求和就是统计为True的过程，再除以本次batch的数量
            acc = torch.sum(prediction == label) / data.shape[0]

            # 计算损失
            loss = loss_func(outputs, label)

            '''
            反向传播三件套
                zero_grad可以将上一次计算得出的梯度清零，因为每次梯度的计算使用的是加法，如果不清0，那么后面梯度的更新就会加入前面计算出来的梯度
                backward反向传播
                step更新参数
            '''
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 进度条旁边打印说明
            processBar.set_description("[%d/%d] Loss : %.8f Acc：%.8f" % (epoch, Epoch, loss, acc))

            # 在最后一轮训练完了以后进行测试
            if index == len(processBar) - 1:
                # 模型在测试之前要加eval，避免好像drop和normalize的影响
                model.eval()
                '''
                    with torch.no_grad()这句话一定要加，很节省显存空间，测试阶段不用计算任何梯度。
                '''
                with torch.no_grad():
                    total_loss = 0.
                    total_right = 0
                    for index, (t_data, t_label) in enumerate(test_loader):
                        # 以下这些和训练的时候一样，可以看上面的训练
                        t_data = t_data.to(device)
                        t_label = t_label.to(device)

                        t_outputs = model(t_data)

                        loss = loss_func(t_outputs, t_label)
                        t_prediction = torch.argmax(t_outputs, dim=1)

                        total_loss += loss
                        total_right += torch.sum(t_prediction == t_label)
                    average_loss = total_loss / test_length
                    total_acc = total_right / test_length
                    # print(average_loss.item())
                    history['Test Loss'].append(average_loss.item())
                    history['Test Accuracy'].append(total_acc.item())
                    processBar.set_description(
                        "[%d/%d] TEST: average_loss %.8f Total_acc %.8f" % (epoch, Epoch, average_loss, total_acc))

        processBar.close()
        torch.save(model.state_dict(), "Simple_CNN.pt")



        torch_model = SimpleM()  # 由研究员提供python.py文件
        torch_model.load_state_dict(torch.load(os.getcwd() + "/Simple_CNN.pt"))
        print(type(torch_model))
        batch_size = 1  # 批处理大小
        input_shape = (3, 224, 224)  # 输入数据

        # set the model to inference mode
        torch_model.eval()

        x = torch.randn(batch_size, *input_shape)  # 生成张量
        export_onnx_file = "Simple_CNN.onnx"  # 目的ONNX文件名
        torch.onnx.export(torch_model,
                          x,
                          export_onnx_file,
                          opset_version=10,
                          do_constant_folding=True,  # 是否执行常量折叠优化
                          input_names=["input_shape"],  # 输入名
                          output_names=["output"]  # 输出名
                          )
