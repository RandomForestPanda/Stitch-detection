import streamlit as st
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

st.header("Stitch Pattern Recognition")

# File uploader for image
uploaded_file = st.file_uploader("Upload a file")

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=16 * 8 * 8, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x

# Load the pre-trained model
model = CNN()
model.load_state_dict(torch.load('custom_model.pth'))
model.eval()  # Set model to evaluation mode

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        print()
        print("class confidence is: {if <0.5---> kasuti bcz we have assumed kasuti stitch as class 0 \n if >0.5---> chikankari bcz we have assumed chikankari stitch as class 1")
        print(output)
        print()
        predicted = torch.where(output > 0.5, torch.tensor(1), torch.tensor(0))

    # Display prediction
    predicted_class = 'KASUTI STITCH' if predicted == 0 else 'CHIKANKARI STITCH'
    if predicted_class=='KASUTI STITCH':
        print("kasuti\n\n\n") #check
    else:
        print("chikankari\n\n\n")
    st.write(f'Predicted class: {predicted_class}')
else:
    st.write("Please upload a supported file format.")
