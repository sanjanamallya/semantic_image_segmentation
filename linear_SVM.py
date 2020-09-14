#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('c:\\users\\sanja\\anaconda\\envs\\pytorch_cuda\\lib\\site-packages')
import ortools
from ortools.linear_solver import pywraplp
import numpy as np
import time
NUM_CLASSES = 2
NUM_FEATURES = 100
NUM_PIXELS = 32*32
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt


# In[ ]:



class VOCDataset(Dataset):
    """Class to store VOC semantic segmentation dataset"""

    def __init__(self, image_dir, label_dir, file_list):

        self.image_dir = image_dir
        self.label_dir = label_dir
        reader = open(file_list, "r")
        self.files = []
        for file in reader:
            self.files.append(file.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 0 stands for background, 1 for foreground
        labels = np.load(os.path.join(self.label_dir, fname+".npy"))
        labels[labels > 0.0] = 1.0
        image = Image.open(os.path.join(self.image_dir, fname+".jpg"), "r")
        sample = (TF.to_tensor(image), torch.LongTensor(labels))

        return sample

class AlexNet(nn.Module):
    """Class defining AlexNet layers using for the convolutional network"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class FCNHead(nn.Sequential):
    """Class defining FCN (fully convolutional network) layers"""

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class SimpleSegmentationModel(nn.Module):
    """
    Class defining end-to-end semantic segmentation model.
    It combines AlexNet and FCN layers with interpolation for deconvolution.
    This model is pretrained using cross-entropy loss.
    After pre-training, use the get_repr() function to construct 32x32x100 feature tensors for each image
    """

    def __init__(self, n_feat, n_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.n_feat = n_feat
        self.backbone = AlexNet()
        self.classifier = FCNHead(256, n_feat)
        self.linear = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, self.n_feat)
        x = self.linear(x)

        return x

    def get_repr(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x


# In[ ]:



class LinearSVM(nn.Module):

    def __init__(self, n_feat, n_classes):
        super(LinearSVM, self).__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        # TODO: Define weights for linear SVM
        self.lin = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        # TODO: Define forward function for linear SVM
        out = self.lin(x)
        return out


# In[ ]:


def train_linear_svm(cnn_model, svm_model, criterion, train_batches, test_batches, device, num_epochs):
    # TODO: Write a training loop for the linear SVM
    # Keep in mind that the CNN model is needed to compute features, but it should not be finetuned
    cnn_model.eval()
    svm_model.train()
    train_loss_list = list()
    test_loss_list =list()
    optimizer = optim.Adam(svm_model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            features = cnn_model.get_repr(images)
            output = svm_model(features)            
            output = output.contiguous().view(-1, 2).squeeze()
            labels = labels.contiguous().view(-1, 1).squeeze()
            loss = criterion.forward(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Training loss after epoch {}: {}".format(epoch, total_loss/len(train_batches)))
        train_time_loss=  total_loss/len(train_batches)
        train_loss_list.append(train_time_loss)
        test_loss = test_linear_svm(cnn_model, svm_model, criterion, train_batches,device)
        test_loss_list.append(test_loss)


    return svm_model, criterion, train_loss_list, test_loss_list


# In[ ]:



def test_linear_svm(cnn_model, svm_model, criterion,test_batches,device):
    # TODO: Write a testing function for the linear SVM
    cnn_model.eval()
    svm_model.eval()
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    total_loss= 0.0
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        features = cnn_model.get_repr(images)
        output = svm_model(features) 
        output_loss_comp = output.contiguous().view(-1, 2).squeeze()
        labels_loss_comp = labels.contiguous().view(-1, 1).squeeze()
        loss = criterion.forward(output_loss_comp, labels_loss_comp)
        total_loss += loss.item()
#         print("models", output.shape)
        output = output.contiguous().squeeze().view(-1, 2)
#         print("Modified", output.shape)
        _, output = torch.max(output, axis=1)
#         print("Maxed", output.shape)
#         visualize_grayscale_image(features.view(32, 32).detach().numpy(), i)
        output = output.squeeze().detach().cpu().numpy()
        labels = labels.contiguous().view(-1, 1).squeeze().cpu().numpy()
        cur_class_pred = np.unique(output, return_counts=True)
        for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
            class_pred[key] += val
        cur_class_gold = np.unique(labels, return_counts=True)
        for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
            class_gold[key] += val
        cur_correct = (output == labels).tolist()
        for j, val in enumerate(cur_correct):
            if val:
                class_correct[labels[j]] += 1
        correct += np.sum(cur_correct)
        total += len(labels)
    class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
    mean_iou = sum(class_iou) / len(class_correct)
    print("Mean IOU: {}".format(mean_iou))
    print("Pixel Accuracy: {}".format(correct / total))
    print("Test time loss: {}".format(total_loss/len(test_batches)))
    
    return (total_loss/len(test_batches))


# In[ ]:


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    #defining paths
    path_to_image_folder = r"C:\Users\sanja\Downloads\DownsampledImages"
    path_to_label_folder= r"C:\Users\sanja\Downloads\DownsampledLabels"
    file_with_train_ids = r"train.txt"
    file_with_test_ids =  r"test.txt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """train and test data loader"""
    train_dataset = VOCDataset(path_to_image_folder, path_to_label_folder, file_with_train_ids)
    test_dataset = VOCDataset(path_to_image_folder, path_to_label_folder, file_with_test_ids)

    train_batches = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    print( "___________CNN__________")
    """Run the cnn model"""
    cnn = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES).to(device)
    train_cnn(cnn, train_batches, test_batches, 1, device)
    """Save the cnn model"""
    torch.save(cnn.state_dict(), 'cnn_model.pth')
    
    # TODO: Instantiate a linear SVM and call train/ test functions
    print( "___________Linear SVM__________")
    svm = LinearSVM(NUM_FEATURES, NUM_CLASSES).to(device)
    criterion = nn.MultiMarginLoss(weight=torch.Tensor([1, 4]).to(device))
    svm, criterion, train_loss_list, test_lost_list = train_linear_svm(cnn, svm,criterion, train_batches, test_batches,device, 1)
    print("Linear svm test results")
    test_linear_svm(cnn, svm, criterion, test_batches, device)
    
    #Plot the loss
    plt.plot(train_loss_list, label = "Train_loss")
    plt.plot(test_lost_list, label = 'Test_loss')
    plt.savefig("linear_svm_loss.png")

    #Save the linear SVM Model
    torch.save(svm.state_dict(), 'svm_model.pth')

    #Visualize Linear SVM results
    features = cnn.get_repr(sample)
    svm_output_ = svm(features)
    svm_output_ = svm_output_.contiguous().squeeze().view(-1, 2)
    _, svm_output = torch.max(svm_output_, axis=1)
    visualize_grayscale_image(svm_output.view(32, 32).detach().cpu().numpy(), 1)
    
if __name__ == "__main__":
    main()

