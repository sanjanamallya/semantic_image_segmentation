#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
NUM_CLASSES = 2
NUM_FEATURES = 100
NUM_PIXELS = 32*32
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt


# In[2]:


def visualize_grayscale_image(image, file=None):
    plt.imshow(image, cmap="gray")
    plt.savefig(str(file)+".png")


# In[3]:


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

        return sample[:5]


# In[4]:



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



# In[5]:



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


# In[6]:



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


# In[7]:


def train_cnn(model, train_batches, test_batches, num_epochs, device):
    """
    This
    function runs a training loop for the FCN semantic segmentation model
    """
    model.train()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 4]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            labels = labels.contiguous().view(-1, 1).squeeze()
            images, labels= images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Training loss after epoch {}: {}".format(epoch, total_loss/len(train_batches)))
        test_cnn(model, test_batches, device)


def test_cnn(model, test_batches, device ):
    """
        This function evaluates the FCN semantic segmentation model on the test set
    """
    model.eval()
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        _, output = torch.max(output, axis=1)
#         print("labels shape", labels.shape)
        output = output.squeeze().detach().cpu().numpy()
        labels = labels.contiguous().view(-1, 1).squeeze().cpu().numpy()
#         print("labels shape after", labels.shape, labels )
        
        cur_class_pred = np.unique(output, return_counts=True)
#         print(cur_class_pred)
        
        for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
            class_pred[key] += val
            
        cur_class_gold = np.unique(labels, return_counts=True)
        
        for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
            class_gold[key] += val
            
        cur_correct = (output == labels).tolist()
#         print(cur_correct, "jj")
        
        for j, val in enumerate(cur_correct):
            if val:
                class_correct[labels[j]] += 1
                
        correct += np.sum(cur_correct)
        total += len(labels)
        
    class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
    mean_iou = sum(class_iou) / len(class_correct)
    print("Mean IOU: {}".format(mean_iou))
    print("Pixel Accuracy: {}".format(correct / total))


# In[8]:


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

    train_batches = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    print( "___________CNN__________")
    """Run the cnn model"""
    cnn = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES).to(device)
    train_cnn(cnn, train_batches, test_batches, 50, device)
#     test_cnn(cnn, test_batches, device)

    """Save the cnn model"""
    torch.save(cnn.state_dict(), 'cnn_model.pth')
#     cnn_model = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES)
#     cnn_model.load_state_dict(torch.load('cnn_model.pth'))

    """ Choose Image for visualization """
    image = Image.open(r"C:\Users\sanja\Downloads\DownsampledImages\2007_003580.jpg")
    sample = (TF.to_tensor(image)).unsqueeze(0).to(device)
    
    """ Visualize CNN output """
    cnn_features =  cnn(sample)
    _, cnn_output = torch.max(cnn_features, axis=1)
#     print(" Output max", cnn_output.shape)
    visualize_grayscale_image(cnn_output.view(32, 32).detach().cpu().numpy(), 0)
    
if __name__ == "__main__":
    main()


# In[9]:


import numpy as np
image = Image.open(r"C:\Users\sanja\Downloads\DownsampledImages\2007_003580.jpg")
data = np.load(r"C:\Users\sanja\Downloads\DownsampledLabels\2007_003580.npy")
np.set_printoptions(threshold=np.inf)
w, h = 32, 32
data = data>0
img = Image.fromarray(data)
display(img)
display(image)


# In[ ]:




