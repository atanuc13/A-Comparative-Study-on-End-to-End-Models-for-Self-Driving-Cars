import csv
import os
import numpy as np
import torch 
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.optim as optim
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

columns = ["img","steering"]
csv_file = '/home/choyya/data_ABIDA/driving1.csv'
image_dir ='/home/choyya/data_ABIDA/data'
print("HIIIIIIIIIIIIIIIIIIIII")
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the custom dataset
#dataset = CustomDataset(image_dir, csv_file, transform=transform)


#########################################################



class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data = []

        with open(self.csv_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip the header
            for line in reader:
                image_name, label = line
                image_path = os.path.join(self.image_dir, image_name)
                self.data.append((image_path, float(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ResNeXt-50 model
import torch.nn as nn
import torchvision.models as models



# Load the pre-trained ResNeXt-50-32x4d model
model = models.resnext50_32x4d(pretrained=True)
#resNext20################################################################################################################
# Modify the model to have fewer layers
#model.layer1 = nn.Sequential(*list(model.layer3.children())[:3])  # remove last three blocks
#model.layer2 = nn.Sequential(*list(model.layer4.children())[:3])  # remove last three blocks
#model.layer3 = nn.Sequential(*list(model.layer3.children())[:2])  # remove last three blocks
#model.layer4 = nn.Sequential(*list(model.layer4.children())[:2])  # remove last three blocks
#model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # change the average pooling layer to adaptive
##########################################################################################################################






#resNext14#################################################################################################################
# Modify the model to have fewer layers
model.layer3 = nn.Sequential(*list(model.layer3.children())[:2])  # remove last two blocks
model.layer4 = nn.Sequential(*list(model.layer4.children())[:2])  # remove last two blocks
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # change the average pooling layer to adaptive
###########################################################################################################################
# Replace the last fully-connected layer with a new one for our specific task
num_classes = 100
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)






# Print the model architecture
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

model = model.to(device)
# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the preprocessing transform
transform = transforms.Compose([
    transforms.Resize((66,200)),
    #transforms.CenterCrop((100, 200)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       # std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(image_dir, csv_file, transform=transform)

# Load the custom dataset
#dataset = CustomDataset(data_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(train_size)
# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=False)

# Train the model
num_epochs = 100
min_run_loss=100.00
min_val_loss=1000.00
train_loss=0.0
val_loss=0.0
val=0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.long()
        # Zero the gradients
        optimizer.zero_grad()
        labels = torch.tensor(labels).view(-1, 1)
        #print("Size of labels: ", labels.size())
        #print("Size of inputs: ", inputs.size())
        # Forward pass
        labels = labels.float()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print the loss every 2000 mini-batches
        #running_loss += loss.item()
        #if i % 200 == 199:
        #print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss))
        #running_loss = 0.0
        
        
        # Update the running training loss
        running_loss += loss.item() * inputs.size(0)

    # Calculate the average training loss for this epoch
    train_loss = running_loss / len(train_dataset)
    if(train_loss<min_run_loss):
      min_run_loss=train_loss
    # Initialize the running validation loss for this epoch
    val_loss = 0.0
    
        # Validate the model
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # Move the images and labels to the device
            #images = images.to(device)
            #labels = labels.to(device)
            inputs, labels = data
            # Forward pass
            labels = torch.tensor(labels).view(-1, 1)
            labels = labels.float()
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the running validation loss
            val=val+loss.item()
            #print(val)
            #val=0
            val_loss += loss.item() * inputs.size(0)

    # Calculate the average validation loss for this epoch
    val_loss = val_loss / len(val_dataset)
    if(val_loss<min_val_loss):
      torch.save(model.state_dict(), "ResNext50_2remove_udacity_100.pth")
      min_val_loss=val_loss
    # Print the loss values
    print(val/len(val_dataset))
    print('Epoch [{}/100], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(
        epoch+1, train_loss, val_loss))


    
print('Minimum training loss:{:.8f} , Validation Loss: {:.6f}'  .format(
         min_run_loss, min_val_loss))  
    
print("Finished training.")

# Eval