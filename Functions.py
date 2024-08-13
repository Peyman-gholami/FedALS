from typing import Dict, List
from scipy.optimize import fsolve
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import json
from bitsandbytes.optim import Adam8bit
from transformers import AdamW
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def learning_rate(t, exp, cte):
    if cte==True:
        return 2**-exp
    #return #1 / (t / 10 + 10)
    return (2**-exp) / (t /cte[0] + cte[1])
    if t<cte[0]:
        return 2 ** -exp
    if cte[0]<=t<cte[1]:
        return (2**-exp)/cte[2]
    return 2**-exp/cte[2]/cte[2]


def softmax(u):
    expu = np.exp(u)
    return expu / np.sum(expu)


def matrix_softmax(u):
    expu = np.exp(u)
    return expu / np.sum(expu, axis=1).reshape(u.shape[0], 1)


def logistic_regression(identity, weight, feature, label, lr, optimizer, device, criterion,tokenizer):
    if identity[0] == "LLM":
        update_lr(optimizer, lr)
        with torch.cuda.amp.autocast():
            batch = tokenizer(feature, truncation=True, padding=True, max_length=128, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            # Forward pass
            out = weight.forward(**batch, )
            one_loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                   reduction='mean')
            # Backward and optimize
            one_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print("loss:", one_loss.item())
        return weight, None
    else:
        update_lr(optimizer, lr)
        if True:#with torch.autocast(device_type='cuda', dtype=torch.float16):
            images = feature.to(device)
            labels = label.to(device)
            # Forward pass
            outputs = weight(images)
            # print(outputs)
            # print(labels)
            one_loss = criterion(outputs, labels)
            # one_loss = Variable(one_loss, requires_grad=True)
            # print(one_loss)
            # input("right?")
            # Backward and optimize
            optimizer.zero_grad()
            one_loss.backward()
            optimizer.step()
            # print("loss:", one_loss.item())
            return weight, None





def loss(identity, data, weight, criterion,device,test_loader,tokenizer,acc):
    if identity[0] == "LLM":
        l_train =0
        l_test = 0
        weight.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for d in data:
                    batch = tokenizer(d['text'], truncation=True, padding=True, max_length=128, return_tensors='pt')
                    batch = {k: v.cuda() for k, v in batch.items()}
                    # Forward pass
                    out = weight.forward(**batch, )
                    one_loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2),
                                               batch['input_ids'][:, 1:].flatten(),
                                               reduction='mean')
                    l_train += one_loss.item()
                for d in test_loader:
                    batch = tokenizer(d['text'], truncation=True, padding=True, max_length=128, return_tensors='pt')
                    batch = {k: v.cuda() for k, v in batch.items()}
                    # Forward pass
                    out = weight.forward(**batch, )
                    one_loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2),
                                               batch['input_ids'][:, 1:].flatten(),
                                               reduction='mean')
                    l_test += one_loss.item()
            # print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        return (l_train/len(data),l_test / len(test_loader))
    else:
        l =0
        weight.eval()
        with torch.no_grad():
            if True:#with torch.autocast(device_type='cuda', dtype=torch.float16):
                for (images, labels) in data:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = weight(images)
                    one_loss = criterion(outputs, labels)
                    l += one_loss.item()
            if acc:
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = weight(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                # print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
                return (l/len(data),correct / total)
            else:
                l_test = 0
                if True:  # with torch.autocast(device_type='cuda', dtype=torch.float16):
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = weight(images)
                        one_loss = criterion(outputs, labels)
                        l_test += one_loss.item()
                return (l / len(data), l_test / len(test_loader))

def aggrigate(models,ws):
    sd=[]
    for model in models:
        sd.append(model.state_dict())
    result_sd = copy.deepcopy(sd[0])
    # Aggrigate all parameters
    for key in sd[0]:
        if 'format' in key:
            continue
        type_of_tensor = sd[0][key]
        result_sd[key] = sd[0][key].float() * ws[0]
        for i in range(1,len(models)):
            result_sd[key] += sd[i][key].float()* ws[i]
        result_sd[key] = result_sd[key].to(type_of_tensor)
    # result_model = copy.deepcopy(models[0])
    # result_model.load_state_dict(result_sd)
    return result_sd

def differ(models,final,ws,layers):
    models_sd=[]
    for model in models:
        models_sd.append(model.state_dict())
    final_sd = final.state_dict()
    d=[0 for layer in layers]
    # Computing norm of difference
    for i in range(len(layers)):
        layer = layers[i]
        count=0
        for sub_layer in layer:
            count += sub_layer[1]
            for j in range(1, len(models)):
                temp = models_sd[j][sub_layer[0]] - final_sd[sub_layer[0]]
                d[i] += (torch.norm(temp.float(),p=2)**2).item() * ws[j]
        print(count)
        d[i] = d[i]/ count
    return d

# ResNet
##########################
### MODEL
##########################


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            try:
                x = Image.open(self.samples[idx]).convert("RGB")
            except:
                x = cv2.cvtColor(cv2.imread(self.samples[idx]), cv2.COLOR_BGR2RGB)
                x = Image.fromarray(x)
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def form_training_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    class_label = ["entailment", "neutral", "contradiction"][example["label"]]
    example[
        "text"
    ] = f"mnli hypothesis: {hypothesis} premise: {premise} target: {class_label}<|endoftext|>"
    genre_dict = {"government": 0 , "fiction": 1, "travel": 2, "slate": 3, "telephone":4, "letters": 5, "verbatim":6,
                  "facetoface": 7, "oup": 8, "nineeleven": 9, }
    example["genre"] = genre_dict[example["genre"]]
    return example

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out, dim=1)
        return out

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class DummyDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y):
        'Initialization'
        self.y = y
        self.x = x

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.x[index], self.y[index]

    def update_label(self, new_labels):
        self.y = new_labels

    def update_feature(self, new_features):
        self.x = new_features