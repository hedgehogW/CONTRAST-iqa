from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision import transforms


class MultiDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # csv文件的行数=df的行数=数据集数量
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 1:].values
        label = label.astype('double')
        # add 把左上角裁剪掉->把左边一杠裁剪掉
        width, height = image.size
        rightline = width / 8.5
        underline = height / 7.5
        img_mask = Image.new("RGB", (int(rightline) + 1, int(underline) + 1))
        image.paste(img_mask, (0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label



def test_dataset():
    root = '/Users/momochan/chapter2/多标签分类/实验数据集/images'
    txt = '/Users/momochan/chapter2/多标签分类/代码/wsdan基础上修改/multi_dataset/train_label.csv'
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }
    multiData = MultiDataset(txt, root, data_transform['train'])
    # print(multiData.num_classes)
    dataloader = DataLoader(multiData, batch_size=16, shuffle=True)
    for data in dataloader:
        images, labels = data
        print(images.size(),labels.size(),labels)


if __name__ == '__main__':
    test_dataset()
