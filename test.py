import os
import csv
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from model import AlexNet


# Set up the data directory and file paths
data_dir = 'data'
csv_file = 'predictions.csv'

import csv
rows = []
with open("data/test_list.txt", 'r') as file:
    rows = file.read().splitlines() 


model = AlexNet()
# load the best model checkpoint
best_model_cp = torch.load('outputs/best_model.pt')
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")

model.load_state_dict(best_model_cp['model_state_dict'])

# Set up the transformation pipeline for the test images
transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get a list of all the test image file names in ascending order
#file_names = os.listdir(os.path.join(data_dir, 'test'))

# Create an empty list to store the predictions
predictions = []

# Loop over the test images and make predictions
model.eval()
print('Testing')

with torch.no_grad():
    for file_name in rows:
        image = Image.open(os.path.join(data_dir, 'test', file_name))
        image_tensor = transform(image)
        output = model(image_tensor.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        predicted = "a1_"+ str(predicted.item())
        print('predicted ',file_name)
        predictions.append((file_name, predicted))

# Write the predictions to a CSV file sorted by image names
with open(csv_file, 'w') as file:
    writer = csv.writer(file)
    for prediction in (predictions):
        print(prediction)
        writer.writerow(prediction)
