import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import time
from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def test(model, testloader, rank):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def train(rank, world_size, epochs=4):
    setup(rank, world_size)
    
    # 数据加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = DataLoader(trainset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化
    model = ResNet50(num_classes=10).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # 计时变量
    total_data_loading_time = 0
    total_training_time = 0
    total_testing_time = 0
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_sampler.set_epoch(epoch)
        
        epoch_data_loading_time = 0
        epoch_training_time = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # 数据加载计时
            data_loading_start = time.time()
            inputs, targets = inputs.to(rank), targets.to(rank)
            data_loading_end = time.time()
            epoch_data_loading_time += data_loading_end - data_loading_start
            
            # 训练计时
            training_start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_end = time.time()
            epoch_training_time += training_end - training_start

        total_data_loading_time += epoch_data_loading_time
        total_training_time += epoch_training_time
        
        # 测试阶段 (只在rank 0进行)
        if rank == 0:
            testing_start = time.time()
            test_acc = test(model, testloader, rank)
            testing_end = time.time()
            epoch_testing_time = testing_end - testing_start
            total_testing_time += epoch_testing_time
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Data Loading Time: {epoch_data_loading_time:.2f}s")
            print(f"  Training Time: {epoch_training_time:.2f}s")
            print(f"  Testing Time: {epoch_testing_time:.2f}s")
            print(f"  Test Accuracy: {test_acc:.2f}%")
        
        scheduler.step()
    
    if rank == 0:
        print("\nFinal Time Summary:")
        print(f"Total Data Loading Time: {total_data_loading_time:.2f}s")
        print(f"Total Training Time: {total_training_time:.2f}s")
        print(f"Total Testing Time: {total_testing_time:.2f}s")
        print(f"Total Execution Time: {time.time() - begin_time:.2f}s")

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs available: {world_size}")
    begin_time = time.time()
    torch.multiprocessing.spawn(train, args=(world_size,4), nprocs=world_size)