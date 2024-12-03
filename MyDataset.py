import csv
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def read_csv_file(file_path):
    data_dict = {}
    # 检查文件是否存在
    if os.path.exists(file_path):
        with open(file_path, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                key = row[0]
                value = int(row[1])
                data_dict[key] = value
    else:
        print(f"File '{file_path}' does not exist. Creating a new file.")
        with open(file_path, mode='w', newline='') as file:
            pass  # 创建一个空白的CSV文件
    return data_dict

class MyDataset(Dataset):
    def __init__(self, csv_name, resize, model):
        super(MyDataset, self).__init__()
        self.csv_name = csv_name
        self.resize = resize
        self.model = model
        self.name2label = {}  # {"name": label}
        self.positive_images, self.negative_images, self.labels = self.load_to_csv(csv_name)
    def load_to_csv(self, csv_name):  # write to current directory
        positive_images, negative_images, labels = [], [], []
        with open(os.path.join(csv_name)) as f:
            reader = csv.reader(f)
            for row in reader:
                negative_img, label = row
                label = int(label)
                parts = negative_img.split('_')
                if 'distortion' in self.csv_name:
                    parts[-3] = parts[-3].replace('0', '1')
                else:
                    parts[-2] = parts[-2].replace('0', '1')
                positive_img = '_'.join(parts)
                if os.path.exists(positive_img) and os.path.exists(negative_img):
                        negative_images.append(negative_img)
                        positive_images.append(positive_img)
                        labels.append(label)
        assert len(positive_images) == len(negative_images) == len(labels)
        return positive_images, negative_images, labels  # return to __init__()

    def __len__(self):
        return len(self.positive_images)

    def __getitem__(self, idx):
        positive_img, negative_img, label = self.positive_images[idx], self.negative_images[idx], self.labels[idx]

        if self.model == 'train':
            transform = transforms.Compose([
                # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.Resize((int(self.resize), int(self.resize))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self.model == 'val':
            transform = transforms.Compose([
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        positive_image = Image.open(positive_img).convert("RGB")  # Abspath ->  PIL image
        negative_image = Image.open(negative_img).convert("RGB")  # Abspath ->  PIL image

        positive_image = transform(positive_image)
        negative_image = transform(negative_image)

        label = torch.tensor(label)

        return positive_image, negative_image, label, negative_img

def test():
    negative_img = "D:\\dataset\\补充发送\\12-河北巨鹿 杭白菊\\HB_JL_HBJ_0_0150.jpg"
    parts = negative_img.split('_')
    parts[-2] = parts[-2].replace('0', '1')
    positive_img = '_'.join(parts)
    print(positive_img)

#test()