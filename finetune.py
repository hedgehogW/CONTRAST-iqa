import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.my_multi_dataset import MultiDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss, accuracy

def main(args):
    # # 加载预训练的对比学习模型
    # model = torch.load('contrastive_learning_model.pth')

    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)

    # 创建新的数据加载器
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    data_dir= args.data_path
    ## 加载训练集
    multiData_train = MultiDataset(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'TrainDataset'), data_transforms['train'])
    dataloader_train = DataLoader(multiData_train, batch_size=224, shuffle=True)
    # dataset_sizes_train = len(multiData_train)

    ## 加载验证集
    multiData_val = MultiDataset(os.path.join(data_dir, 'val.csv'), os.path.join(data_dir, 'ValDataset'), data_transforms['val'])
    dataloader_val = DataLoader(multiData_val, batch_size=224, shuffle=True)
    # dataset_sizes_val = len(multiData_val)

    # 修改模型头部
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3是你任务的类别数量

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    num_epochs = 300
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, dataloader_train, criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloader_val, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        print("-" * 20)

    print("Training complete")
    torch.save(model.state_dict(), 'your_model_name.pth')



def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--im_path', type=str, \
    #                     default='sample_images/33.bmp', \
    #                     help='Path to image', metavar='')
    parser.add_argument('--model_path', type=str, \
                        default='models/CONTRIQUE_checkpoint25.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    # parser.add_argument('--csv_path',type=str, \
    #                     default='dataset/train.csv', \
    #                     help='',metavar='')
    # parser.add_argument('--stage',type=str, \
    #                     default='train', help='train/val/test',metavar='')
    # parser.add_argument('--linear_regressor_path', type=str, \
    #                     default='models/CLIVE.save', \
    #                     help='Path to trained linear regressor', metavar='')
    parser.add_argument('--data_path',type=str, \
                        default='dataset', \
                        help='Path to Dataset of Training',metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)