import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision.transforms import v2

import torch
import torch.nn as nn
import torch.nn.functional as F


class Preprocessing:
    def __init__(self):
        self.conversions = {'top': 1.0, 'verstop': 0.75, 'vers': 0.5, 'versbottom': 0.25, 'bottom': 0.0}

        self.x_img = None
        self.x_features = None
        self.y = None

    def preprocess_img(self, img):
        '''Converts a PIL image to a 2D array'''
        img = img.convert("L")                                          # grayscale

        square_length = min(img.size)
        img = v2.CenterCrop(size=(square_length, square_length))(img)   # center crop
        img = v2.Resize(size=(212,212))(img)                            # resize
        img = np.array(img)
        return img
    
    def label2value(self, label):
        value = self.conversions[label]
        return value
    
    def value2label(self, value):
        conversions2 = {'Top': 1.0, 'Vers Top': 0.75, 'Versatile': 0.5, 'Vers Bottom': 0.25, 'Bottom': 0.0}
        label = min(conversions2, key = lambda k: abs(conversions2[k] - value))

        return label
    
    def extract_features(self, filepath):
        features = filepath.split('/')[-1].split('_')
        label, height, weight = features[0], features[1], features[2]
        height, weight = float(height), float(weight)
        return label, height, weight
    
    def data2tensors(self, filepaths):
        '''For creating train/test sets'''
        x_img, x_features, y = [], [], []

        for fp in filepaths:
            try:
                # Processing images
                img = Image.open(fp)
                img = self.preprocess_img(img)
                x_img.append(img)

                # Processing features
                label, height, weight = self.extract_features(fp)
                x_features.append((height, weight))

                # Process label
                label = self.label2value(label)
                y.append(label)
                
            except Exception as e:
                print(fp)
                print(e)
                print()

        assert len(x_img) == len(x_features) == len(y), f"inconsistent lengths: X_img ({len(x_img)}), X_features ({len(x_features)}), y ({len(y)})"

        x_img, x_features, y = np.array(x_img), np.array(x_features), np.array(y)
        x_img, x_features, y = torch.tensor(x_img, dtype=torch.float32), torch.tensor(x_features, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        # x_img = x_img.unsqueeze(1)

        return x_img, x_features, y
    
    def preprocess_instance(self, img, height, weight):
        pass



# ————————————————————————————————————————————————————————————————————————————————————————————


class CNNWithAdditionalFeatures(nn.Module):
    def __init__(self):
        super(CNNWithAdditionalFeatures, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=258, kernel_size=3, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        
        # Additional feature input
        self.fc_extra = nn.Linear(2, 20)  # Expecting 2 extra features
        
        # Corrected fc1 input size based on (batch_size, 1, 212, 212)
        self.fc1 = nn.Linear(31218, 100)
        self.fc2 = nn.Linear(100 + 20, 100)  # Merge CNN and extra features
        self.fc3 = nn.Linear(100, 100)
        self.fc_out = nn.Linear(100, 1)

    def forward(self, image, extra_features):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Process additional features
        extra_features = F.relu(self.fc_extra(extra_features))
        
        # Merge both branches
        x = torch.cat((x, extra_features), dim=1)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc_out(x)
        return x
    


# class CNNWithAdditionalFeatures(nn.Module):
#     def __init__(self):
#         super(CNNWithAdditionalFeatures, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=258, kernel_size=3, padding=0)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.25)
        
#         # Additional feature input
#         self.fc_extra = nn.Linear(2, 20)  # Expecting 2 extra features
        
#         # Corrected fc1 input size (Placeholder: will be validated with prints)
#         self.fc1 = nn.Linear(31158, 200)  # Adjust if needed
#         self.fc2 = nn.Linear(200 + 20, 200)  # Merge CNN and extra features
#         self.fc3 = nn.Linear(200, 200)
#         self.fc4 = nn.Linear(200, 200)
#         self.fc_out = nn.Linear(200, 1)

#     def forward(self, image, extra_features):
#         print(f"Input image shape: {image.shape}")  # Debugging input shape

#         x = self.pool(F.relu(self.conv1(image)))
#         print(f"After Conv1 + Pool: {x.shape}")  

#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         print(f"After Conv2 + Pool: {x.shape}")  

#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         print(f"After Conv3 + Pool: {x.shape}")  

#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv4(x)))
#         print(f"After Conv4 + Pool: {x.shape}")  

#         x = self.dropout(x)
        
#         x = torch.flatten(x, start_dim=1)
#         print(f"Flattened Shape: {x.shape}")  # This will confirm if 31158 is correct

#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        
#         # Process additional features
#         extra_features = F.relu(self.fc_extra(extra_features))
        
#         # Merge both branches
#         x = torch.cat((x, extra_features), dim=1)
#         x = self.dropout(x)
        
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc4(x))
#         x = self.dropout(x)
        
#         x = self.fc_out(x)
#         return x
