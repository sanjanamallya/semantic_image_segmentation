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
#         self.files = self.files[:5]

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


# In[4]:


class StructSVM(nn.Module):

    def __init__(self, n_feat, n_classes, w, h):
        super(StructSVM, self).__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.w = w
        self.h = h
        # TODO: Define weights for structured SVM        
        self.lin1 = nn.Linear(n_feat, n_classes)
        self.lin2 = nn.Linear( 2 * n_feat, n_classes)


    def forward(self, image, label, phase):

        
        NUM_ROW = self.h
        NUM_COL = self.w
        NUM_CLASS = 2
        NUM_PIXELS = NUM_ROW * NUM_COL
        
        #############################################################
        features = image.contiguous().view(-1, 100).squeeze()
        # image ( 32 x 32 x 100)
        loss_term1 = self.lin1(image)
        "Retrive unary potential for the pixels"
        # loss_term1 ( 32 x 32 x 2)
        #u_p (1024 x 2)
        u_p= loss_term1.contiguous().view(-1, 2)
        
        """ In training phase, we need labels for structured hinge loss and use it in the ILP objective
           during evaluation we use MAP formulation of ILP that doesn't include the loss """
        
        if phase == 'train':
            x_true= label.contiguous().view(-1)
            x_true_oh = torch.nn.functional.one_hot(x_true, num_classes= 2)   
            x_true_oh_detached = x_true_oh.clone().detach().cpu().numpy() 

        
        solver =  pywraplp.Solver('LinearExample', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        # Define assignment variables
        pixel_assignment = {}
        edge_assignment = {}

        ############################################################
        "Defining the variables"
        for pixel in range(NUM_PIXELS):
            for c in range(NUM_CLASS):
                pixel_assignment[pixel, c] = solver.IntVar(0.0, 1.0, 'x[%d,%d]' % (pixel, c))

                # for every pixel in the array, if it is not the last pixel in the row, it will have a right neighbour
                if ((pixel+1)% NUM_COL != 0):
                    edge_assignment[pixel, pixel+1 , c] = solver.IntVar(0.0, 1.0, 'x[%d,%d %d]' % (pixel, pixel+1, c))

                # for every pixel in the array that does not belong to the last row, it will have a bottom neighbour 
                if (pixel+ NUM_COL < NUM_PIXELS):
                    bottom = pixel + NUM_COL
                    edge_assignment[pixel, bottom, c] = solver.IntVar(0.0, 1.0, 'x[%d,%d %d]' % (pixel,bottom, c))
        
        ##############################################################
        "Defining the constraints"

        # Constraint 1: Only one assignment per pixel
        for i in range(NUM_PIXELS):
            solver.Add(solver.Sum([pixel_assignment[i, j] for j in range(NUM_CLASS)]) == 1)
            
        # Constraint 2: Constraints on the edge  
        edge_dictionary = {}
        # maintain an embedding list of the order of the edges for computing the edge potentials later 
        edge_list =list()
        for pixel in range(NUM_PIXELS):
            neighbour_list = list()
            for c in range(NUM_CLASS):
                
                if ((pixel+1)% NUM_COL != 0):
                    right = pixel + 1 
                    if c==0:
                        neighbour_list.append(right)
                        edge_list.append((pixel,right))
                    solver.Add(edge_assignment[pixel, right , c] <= pixel_assignment[pixel,c])
                    solver.Add(edge_assignment[pixel, right , c] <= pixel_assignment[right,c])

                if (pixel+ NUM_COL < NUM_PIXELS):
                    bottom = pixel + NUM_COL
                    if c==0:
                        neighbour_list.append(bottom)
                        edge_list.append((pixel,bottom))

                    solver.Add(edge_assignment[pixel, bottom , c] <= pixel_assignment[pixel,c])
                    solver.Add(edge_assignment[pixel, bottom , c] <= pixel_assignment[bottom,c])
                
            edge_dictionary[pixel]= neighbour_list
            del neighbour_list

        ###########################################################################
        "Do not require initial assignment "
# #         Invoke Solver and  Check that a feasible optimal solution was found
#         result = solver.Solve()   
#         assert result == pywraplp.Solver.OPTIMAL
#         assert solver.VerifySolution(1e-7, True)
        
#         # Gather the optimal assignment
#         initial_pixel_assignment = {} #np.zeros((NUM_PIXELS, NUM_CLASS))
#         initial_edge_assignment = {} #np.zeros((len(edge_list), len(edge_list), NUM_CLASS))
#         for i in range(NUM_PIXELS):
#             for j in range(NUM_CLASS):
#                 initial_pixel_assignment[i,j] = pixel_assignment[i,j].solution_value()
                
#                 if (( i+1)% NUM_COL != 0):
#                     initial_edge_assignment[i, i + 1 , c] = edge_assignment[i, i+1, c].solution_value()

#                 if ( i+ NUM_COL < NUM_PIXELS):
#                     bottom = i + NUM_COL
#                     initial_edge_assignment[i, bottom , c] = edge_assignment[i, bottom , c].solution_value()
         
        ##############################################################################
        """Compute the edge potential"""
        emb_list= list()
        for i in range(len(edge_list)):
            emb = torch.cat((features[edge_list[i][0]], features[edge_list[i][1]]))
            emb_list.append(emb)
        emb_arr = torch.stack(emb_list)
        del emb_list
        
        #pass it through a linear layer to multiply them with weights
        # emb_arr ( 1984 x 200)
        e_p= self.lin2(emb_arr)        
        # e_p ( 1984 x 2)
        # Detach and convert to a numpy array to pass it to the ILP
        u_p_detached = u_p.clone().detach().cpu().numpy()
        e_p_detached = e_p.clone().detach().cpu().numpy()
        
        #################################################################################
        """ Define objective to maximize """
        
        objective = 0.0
        loss_objective = 0.0
        for i in range(NUM_PIXELS):
            for j in range(NUM_CLASS):
                objective += u_p_detached[i,j] * pixel_assignment[i,j]
                
                if ((pixel+1)% NUM_COL != 0):
                    right = pixel + 1 
                    objective += e_p_detached[i,right,j] * edge_assignment[i,right,j]
                    
                if (pixel+ NUM_COL < NUM_PIXELS):
                    bottom = pixel + NUM_COL
                    objective += e_p_detached[i,bottom,j] * edge_assignment[i,bottom,j]
                    
                if phase == 'train':
                    loss_objective += ((1- pixel_assignment[i,j] ) * x_true_oh_detached[i,j]) + ((1- x_true_oh_detached[i,j]) * pixel_assignment[i,j]) 
        
        loss_objective = loss_objective/ (NUM_PIXELS * NUM_CLASS) 
        objective += loss_objective
        solver.Maximize(objective)
        
        #############################################################################
        """Solve ILP with objective """
        
        result = solver.Solve()
        assert result == pywraplp.Solver.OPTIMAL
        assert solver.VerifySolution(1e-7, True)
        final_pixel_assignment = {} 
        final_edge_assignment = {} 
        for i in range(NUM_PIXELS):
                for j in range(NUM_CLASS):
                    final_pixel_assignment[i,j] = pixel_assignment[i,j].solution_value()
                
                if (( i+1)% NUM_COL != 0):
                    for j in range(NUM_CLASS):
                        final_edge_assignment[i, i + 1 , j] = edge_assignment[i, i+1, j].solution_value()

                if ( i+ NUM_COL < NUM_PIXELS):
                    bottom = i + NUM_COL
                    for j in range(NUM_CLASS):
                        final_edge_assignment[i, bottom , j] = edge_assignment[i, bottom , j].solution_value()
        
        return u_p, e_p, final_pixel_assignment, final_edge_assignment 
    


# In[5]:


# TODO: Write a function to compute the structured hinge loss
# using the max-scoring output from the ILP and the gold output
def compute_struct_svm_loss(phase, x_pred, y_pred, x_true, unary_potential, edge_potential, device):
#     print(x_pred.is_cuda, y_pred.is_cuda, x_true.is_cuda, unary_potential.is_cuda, edge_potential.is_cuda)    
    print("Computing Loss")
    NUM_COL = 32
    NUM_CLASS = 2
    NUM_PIXELS = int(len(x_pred)/2)
    edge_len = int(len(y_pred)/2)
    true_edge_assignment = {}
    # x_true (32 x 32)
    x_true= x_true.contiguous().view(-1)
    x_true_oh = torch.nn.functional.one_hot(x_true, num_classes= 2)   
    
    """ edge assignment is one hot encoded and given a value pertaining to the class it is connected to
        If both pixels an edge is touching belong to different class then edge assignment zero """
    for i in range(NUM_PIXELS):
        if ((i+1)% NUM_COL != 0) and (i+1)< 1024:
                true_edge_assignment[i, i + 1 , 0] = 0
                true_edge_assignment[i, i + 1 , 1] = 0
                #Check if neigboring pixels belong to the same class 
                if(x_true[i]== x_true[i+1]):
                    plugin1= int(x_true[i])
                    # Assign that class a value of one
                    true_edge_assignment[i, i + 1 , plugin1] = 1
                
        if ( i+ NUM_COL < NUM_PIXELS):
            bottom = i + NUM_COL
            if bottom < 1024:
                true_edge_assignment[i, bottom , 0] = 0
                true_edge_assignment[i, bottom , 1] = 0
                if(x_true[i]== x_true[bottom]):
                    plugin2 = int(x_true[i])
                    true_edge_assignment[i, bottom , plugin2] = 1
    
    # Convert the predicted pixel and edge assignmnets that are dictionaries into matrices
    predicted_edge_assignments =  torch.from_numpy(np.asarray([*y_pred.values()]).reshape(edge_len, 2)).to(device)
    actual_edge_assignments =  torch.from_numpy(np.asarray([*true_edge_assignment.values()]).reshape(edge_len, 2)).to(device)
    x_pred_oh =  torch.from_numpy(np.asarray([*x_pred.values()]).reshape(NUM_PIXELS, 2)).to(device)
    _, x_pred_output = torch.max(x_pred_oh, axis=1)

    print(unary_potential.is_cuda, x_pred_oh.is_cuda)
    # Compute Score cap
    S_cap_x = torch.sum(unary_potential* x_pred_oh )   
    S_cap_y = torch.sum(edge_potential* predicted_edge_assignments )
    S_cap = S_cap_x + S_cap_y
    
    # Compute Score hat
    S_star_x = torch.sum(unary_potential* x_true_oh) 
    S_star_y = torch.sum(edge_potential* actual_edge_assignments )
    S_star =  S_star_x + S_star_y
    
    #Compute the hamming loss
    term1=  x_pred_oh* (1- x_true_oh )
    term2=  x_true_oh* (1- x_pred_oh )
    ham_loss= torch.sum(term1 +term2) / (NUM_PIXELS* NUM_CLASS)
    
    #Compute hinge loss
    xx =  S_cap - S_star + ham_loss
    L = torch.max(torch.tensor(0,dtype =torch.float64).to(device), xx)  
#     pdb.set_trace()

    return L


# In[6]:



def train_struct_svm(cnn_model, svm_model, train_batches, test_batches, num_epochs, device):
    # TODO: Write a training loop for the structured SVM
    # Keep in mind that the CNN model is needed to compute features, but it should not be finetuned
    total_loss= 0.0
    cnn_model.eval()
    svm_model.train()
    optimizer = optim.Adam(svm_model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch 
            images = images.to(device)
            labels = labels.to(device)
            features = cnn_model.get_repr(images)
            unaryp, edgep, pixel_assignment, edge_assignment= svm_model(features, labels, 'train')        
            loss =  compute_struct_svm_loss('train',pixel_assignment, edge_assignment, labels, unaryp, edgep, device)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Training loss after epoch {}: {}".format(epoch, total_loss/len(train_batches)))
        test_struct_svm(cnn_model, svm_model, test_batches, device)

    return


# In[7]:


def test_struct_svm(cnn_model, svm_model, test_batches, device):
    # TODO: Write a testing function for the structured SVM
    NUM_PIXELS = 32*32
    cnn_model.eval()
    svm_model.eval()
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    total_loss = 0.0
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        features = cnn_model.get_repr(images)
        unaryp, edgep, pixel_assignment, edge_assignment= svm_model(features, labels, 'test') 
        # Pass test flag 
        loss =  compute_struct_svm_loss('test' ,pixel_assignment, edge_assignment, labels, unaryp, edgep, device)
        total_loss+= loss.item()
        xs =  torch.from_numpy(np.asarray([*pixel_assignment.values()]).reshape(NUM_PIXELS, 2))
        _, output = torch.max(xs, axis=1)
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
    print("Test time loss {}".format(total_loss/len(test_batches)))            
    return mean_iou, correct/total, total_loss/len(test_batches)


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

    
    CNN_model = False
    SSVM_MODEL = True 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """train and test data loader"""
    train_dataset = VOCDataset(path_to_image_folder, path_to_label_folder, file_with_train_ids)
    test_dataset = VOCDataset(path_to_image_folder, path_to_label_folder, file_with_test_ids)

    train_batches = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    image = Image.open(r"C:\Users\sanja\Downloads\DownsampledImages\2007_001289.jpg")
    data = np.load(r"C:\Users\sanja\Downloads\DownsampledLabels\2007_001289.npy")
    sample = (TF.to_tensor(image)).unsqueeze(0).to(device)
    
    if CNN_model:
        print( "___________CNN__________")
        """Run the cnn model"""
        cnn = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES).to(device)
        train_cnn(cnn, train_batches, test_batches, 1, device)
        """Save the cnn model"""
        torch.save(cnn.state_dict(), 'cnn_model.pth')
    else:
        cnn = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES).to(device)
        cnn.load_state_dict(torch.load('cnn_model.pth'))
  
        
    print( "___________SSVM__________")

    structured_svm = StructSVM(NUM_FEATURES, NUM_CLASSES, 32, 32).to(device)
    train_struct_svm(cnn, structured_svm, train_batches, test_batches, 3, device )

    #Save the model
    torch.save(structured_svm.state_dict(), 'sd_svm_model.pth')
    
    
    #Visualize the results
    print( "___________CNN_VISUALIZATION__________")    
    features = cnn.get_repr(sample)
    labels= None
    cnn_features = cnn(sample)
    _, cnn_output = torch.max(cnn_features, axis=1)
    visualize_grayscale_image(cnn_output.view(32, 32).detach().cpu().numpy(), 0)

    
    print( "___________SSVM_VISUALIZATION__________")
    features = cnn.get_repr(sample)
    labels= None
    unaryp, edgep, pixel_assignment, edge_assignment= structured_svm(features, labels, 'test')
    x_pred_oh =  torch.from_numpy(np.asarray([*pixel_assignment.values()]).reshape(NUM_PIXELS, 2))
    _, x_pred_output = torch.max(x_pred_oh, axis=1)
    visualize_grayscale_image(x_pred_output.view(32, 32).detach().cpu().numpy(), 2)


if __name__ == "__main__":
    main()


# In[ ]:




