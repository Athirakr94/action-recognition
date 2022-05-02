import numpy as np
import cv2
import torch
import torchvision
import os
import torch
# import torch.nn as nn
# import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import *
from matplotlib import pyplot as plt

X_train= np.load('train_X64_bal.npy')
targets_train= np.load('train_Y64_bal.npy')
X_test= np.load('test_X64_bal.npy')
targets_test= np.load('test_Y64_bal.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train= X_train.reshape(7166,3,7,64,64)
X_test= X_test.reshape(2779,3,7,64,64)

test_x = torch.from_numpy(np.array(X_test)).float()
test_y = torch.from_numpy(np.array(targets_test)).long()
train_x = torch.from_numpy(np.array(X_train)).float()
train_y = torch.from_numpy(np.array(targets_train)).long()
batch_size = 16 #We pick beforehand a batch_size that we will use for the training


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
num_classes = 5

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 16,(1,1,1))
        self.conv_layer2 = self._conv_layer_set(16, 32,(1,1,1))
        self.conv_layer3 = self._conv_layer_set(32, 64,(1,1,1))
        self.conv_layer4 = self._conv_layer_set(64, 128,(2,2,2))
        self.conv_layer5 = self._conv_layer_set(128, 128,(2,2,2))

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
#         self.fc3 = nn.Linear(num_classes, 1)

        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)  
#         self.softmax= nn.Softmax(dim=1)
        
    def _conv_layer_set(self, in_c, out_c,stride):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.LeakyReLU(),
        nn.MaxPool3d(stride),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
#         print("After conv1",out.shape)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        print("After conv5",out.shape)
        out=out.flatten()
        print("After flatten",out.shape)
        out=out.reshape(batch_size,32768)
        print("After flatten",out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.batch(out)
#         print("After fc1",out.shape)
        out = self.relu(out)
        
        out = self.drop(out)
        out = self.fc2(out)
#         print("After fc2",out.shape)
#         out= self.fc3(out)
#         print("After fc3",out.shape)
        

        
        return out

#Definition of hyperparameters
n_iters = 10000
num_epochs = n_iters / (len(train_x) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)

# Create CNN
model = CNNModel()
#model.cuda()
print(model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 1e-4 #0.001
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning
# CNN model training
train_losses=[]
test_losses=[]
epochs=[]
accuracy_list=[]
for epoch in range(10): #I decided to train the model for 50 epochs
    loss_ep = 0
    loss_test=0
    epochs.append(epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data
        targets = targets
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = error(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
#     print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_loader)}")
    train_losses.append(loss_ep/len(train_loader))
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data,targets) in enumerate(test_loader):
            data = data 
            targets = targets
            ## Forward Pass
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            loss = error(scores,targets)
            loss_test+=loss.item()
        accuracy=float(num_correct) / float(num_samples) * 100
        print(
        #     f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}"
        )
        accuracy_list.append(accuracy)
        test_losses.append(loss_test/len(train_loader))
# visualization loss 
# iteration_list=[i for i in range(30)]
plt.plot(epochs,train_losses,label="train loss")
plt.plot(epochs,test_losses,label="test loss")
plt.legend()
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.title("3DCNN: Loss vs Number of epoch")
plt.savefig("deep3d_trainloss.png")
plt.figure()
# visualization accuracy 
plt.plot(epochs,accuracy_list,color = "red")
plt.xlabel("Number of epoch")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of epoch")
plt.savefig("deep3d_train accuracy.png")
torch.save(model.state_dict(), "deep3d_30.pth")
print("Test loss :",loss_test/num_samples)
print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")