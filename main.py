import argparse
import csv
import glob
import os
import random
import torch
import re
from torch import optim, nn
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from tqdm import tqdm
from resnetx import ModifiedResNet18
import matplotlib.pyplot as plt
import numpy as np
import timm
import timm.optim
import timm.scheduler
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
# from TwoStreamSwinT import TwoStreamViT
# from swin_transformer import Net, BaselineModel1,Net2
# from ConvSwin import ConSwinNet2,ConSwinNet2
# from TwoStreamSwinTransformer import TwoStreamSwinTransformer
# from network_swinfusion import TwoStreamSwinTransformer
from MutiScale import TwoStreamSwinTransformer
# from MutiScale2 import TwoStreamSwinTransformer
best_acc = 0
best_epoch = 0
batch_size = 24
epochs = 30
pepochs = 20
nepochs = 20
extend_epochs = 10
num_classes = 19
data_path = "/home/user/dataset3origin/"
matrix_path = "error_matrix.txt"
matrix2_path = "error_matrix19.txt"
pretrained = True
csv_file_path = "error_counts2.csv"
learning_rate = 0.00002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
error_matrix = [[0] * (2) for _ in range(num_classes)]
error_matrix2 = [[0] * (num_classes) for _ in range(num_classes)]

def save_to_txt(error_matrix, file_path):
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            for row in error_matrix:
                file.write(" ".join(map(str, row)) + "\n")
    else:
        old_error_matrix = []
        with open(file_path, "r") as file:
            for line in file:
                row = list(map(int, line.strip().split()))
                old_error_matrix.append(row)
        result = []
        for i in range(len(old_error_matrix)):
            row_result = []
            for j in range(len(old_error_matrix[0])):
                sum_element = old_error_matrix[i][j] + error_matrix[i][j]
                row_result.append(sum_element)
            result.append(row_result)
        # print(error_matrix)
        with open(file_path, "w") as file:
            for row in result:
                file.write(" ".join(map(str, row)) + "\n")

def mixup_data(x1, x2, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# def train(epoch, model, criterion, optimizer, train_loader, val_loader, save_model):
#     global best_acc
#     global best_epoch
#     running_loss = 0.0
#     model.train()
#     # train_loader_with_progress = tqdm(train_loader)
#     for batch_idx, data in enumerate(train_loader, 1):
#         x1,y1,labels, file_name = data
#         x1,y1,labels = x1.to(device),y1.to(device),labels.to(device)
#         outputs = model(x1, y1)
#         optimizer.zero_grad()
#         # print(outputs, labels)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
#         optimizer.step()
#         running_loss += loss.item()
#         # print(loss.item())
#         if batch_idx % 10 == 0:  # 每10个batch输出一次损失
#             print("[%d, %d] loss: %.3f" % (epoch, batch_idx, running_loss % 100))
#             running_loss = 0.0
#     if epoch % 1 == 0:
#         val_acc = test(model, val_loader)
#         if val_acc > best_acc:
#             best_acc, best_epoch = val_acc, epoch
#             torch.save(model.state_dict(), save_model)

#     print("Accuracy Of Val Set:", val_acc * 100.0, "%")
#     print("Best Epoch:", best_epoch, ", Accuracy Of Val Set:", best_acc * 100, "%")

def train(epoch, model, criterion, optimizer, train_loader, val_loader, save_model):
    global best_acc
    global best_epoch
    running_loss = 0.0

    x1_list = []
    y1_list = []
    labels_list = []
    outputs_list = []
    model.train()
    for batch_idx, data in enumerate(train_loader, 1):
        x1, y1, labels, file_name = data
        x1, y1, labels = x1.to(device), y1.to(device), labels.to(device)
        
        # Apply mixup
        mixed_x1, mixed_x2, targets_a, targets_b, lam = mixup_data(x1, y1, labels,0.2)
        outputs = model(mixed_x1, mixed_x2)

        optimizer.zero_grad()
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:  # 每10个batch输出一次损失
            print("[%d, %d] loss: %.3f" % (epoch, batch_idx, running_loss / 10))
            running_loss = 0.0

        x1_list.append(mixed_x1)
        y1_list.append(mixed_x2)
        labels_list.append(labels)
        outputs_list.append(outputs)
        
        if batch_idx % 8 == 0:
            tot = 0
            cor = 0
            new_x = torch.cat(x1_list, dim=0)
            new_y = torch.cat(y1_list, dim=0)
            new_labels = torch.cat(labels_list, dim=0)
            new_outputs = torch.cat(outputs_list, dim=0)

            prob_outputs = F.softmax(new_outputs, dim=1)
            prob_labels = torch.zeros_like(prob_outputs).scatter_(1, new_labels.unsqueeze(1), 1)
            diff = torch.norm(prob_outputs - prob_labels, p=2, dim=1)
            _, top_indices = torch.topk(diff, 24, largest=True)
            top_x = new_x[top_indices]
            top_y = new_y[top_indices]
            top_labels = new_labels[top_indices]
            # print(top_labels)
            
            optimizer.zero_grad()
            top_outputs = model(top_x, top_y)
            _, predicted = torch.max(top_outputs.data, dim=1)
            tot += top_labels.size(0)
            cor += (predicted == top_labels).sum().item()
            loss = criterion(top_outputs, top_labels)
            loss.backward()
            optimizer.step()
            
            x1_list.clear()
            y1_list.clear()
            labels_list.clear()
            outputs_list.clear()

    if epoch % 1 == 0:
        val_acc = test(model, val_loader)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), save_model)

    print("Accuracy Of Val Set:", val_acc * 100.0, "%")
    with open("speed.txt", 'a') as file:
        file.write(str(val_acc)+ "\n")    
    print("Best Epoch:", best_epoch, ", Accuracy Of Val Set:", best_acc * 100, "%")


def test(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            x1, y1, labels, file_name = data
            x1, y1, labels = x1.to(device), y1.to(device), labels.to(device)
            outputs = model(x1, y1)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print("Accuracy on test set : %.3f%%" % (100 * correct / total), " [%d / %d]" % (correct, total))
        return round(correct / total, 3)

def test2(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            x1, y1, labels, file_name = data
            x1, y1, labels = x1.to(device), y1.to(device), labels.to(device)
            outputs = model(x1, y1)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            right_indices = (predicted == labels).nonzero(as_tuple=True)[0]
            # if epoch>20:
            for idx in wrong_indices:
                actual_label = labels[idx].item()
                predicted_label = predicted[idx].item()
                # 根据预测结果和真实标签确定错误的类别
                error_matrix[actual_label][0] += 1
                error_matrix[actual_label][1] += 1
                error_matrix2[actual_label][predicted_label] += 1
            for idx in right_indices:
                actual_label = labels[idx].item()
                predicted_label = predicted[idx].item()
                # 根据预测结果和真实标签确定错误的类别
                error_matrix[actual_label][1] += 1
                error_matrix2[actual_label][predicted_label] += 1
        # print("Accuracy on test set : %.3f%%" % (100 * correct / total), " [%d / %d]" % (correct, total))
        return round(correct / total, 3)


def main(iteration):

    train_csv = '/home/user/dataloader/dataset3origin/'+ f"train_set{iteration}.csv"
    val_csv = '/home/user/dataloader/dataset3origin/'+ f"val_set{iteration}.csv"
    test_csv = '/home/user/dataloader/dataset3origin/'+ f"test_set{iteration}.csv"

    train_dataset = MyDataset(train_csv, 224 ,'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              prefetch_factor=6)
    val_dataset = MyDataset(val_csv, 224,'val')
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            pin_memory=True,
                            prefetch_factor=6)
    test_dataset = MyDataset(test_csv, 224,'val')
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=8,
                             pin_memory=True,
                             prefetch_factor=6)

    model = TwoStreamSwinTransformer(patch_size=4, window_size=7, embed_dim=96, depths=[2, 2, 18, 2],
                                     conv_dims=(96, 192, 384, 768), num_heads=[3, 6, 12, 24],
                                     num_classes=num_classes).to(device)
    pretrained_dict = torch.load('/home/user/model_pth/checkpoint/swin_small_patch4_window7_224_22kto1k.pth',
                                 map_location='cpu')
    model_dict = model.state_dict()
    updated_dict = {f'stream1_{k}': v for k, v in pretrained_dict['model'].items() if f'stream1_{k}' in model_dict}
    updated_dict2 = {f'stream2_{k}': v for k, v in pretrained_dict['model'].items() if f'stream2_{k}' in model_dict}
    model_dict.update(updated_dict)
    model.load_state_dict(model_dict, strict=True)
    model_dict.update(updated_dict2)
    model.load_state_dict(model_dict, strict=True)

    # model= TwoStreamViT(num_classes).to(device)
    # 定义加权交叉熵损失函数

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs+1):
        train(epoch, model, criterion, optimizer, train_loader, val_loader, 'model/'+f"model_fold.pth")
    # for epoch in range(1, epochs+1):
    #     train(epoch, model, criterion, optimizer, train_loader, 'model/'+f"model_fold.pth")
        
    model.load_state_dict(torch.load('model/'+f"model_fold.pth"))
    test_acc = test2(model, test_loader)
    print("Accuracy Of Test Set:", test_acc * 100.0, "%")
    with open("result.txt", 'a') as file:
        file.write(str(test_acc)+ "\n")
    save_to_txt(error_matrix, matrix_path)

    with open("error_matrixep.txt", "a") as file2:
        for row in error_matrix2:
            file2.write(" ".join(map(str, row)) + "\n")
        file2.write("\n")

    save_to_txt(error_matrix2, matrix2_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, validate and test a model.")
    parser.add_argument('--iteration', type=int, required=True, help="Iteration number (1 to 20).")
    args = parser.parse_args()
    main(args.iteration)
# plt.plot(np.arange(epochs),lrr_list)
# plt.show()