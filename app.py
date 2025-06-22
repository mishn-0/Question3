import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# CVAE Model (same as in train_cvae.py)
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(28*28 + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, 28*28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, y):
        h1 = self.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        h3 = self.relu(self.fc3(torch.cat([z, y], dim=1)))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def one_hot(labels, num_classes=10):
    if isinstance(labels, int):
        labels = [labels]
    labels = torch.tensor(labels)
    return torch.eye(num_classes)[labels]

def load_model(path='cvae_model.pth', device='cpu'):
    model = CVAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def generate_images(model, digit, num_images=5, device='cpu'):
    model.eval()
    images = []
    y = one_hot([digit]*num_images).to(device)
    z = torch.randn(num_images, model.latent_dim).to(device)
    with torch.no_grad():
        out = model.decode(z, y).cpu().numpy()
    for i in range(num_images):
        img = out[i].reshape(28, 28)
        images.append(img)
    return images

def show_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)

st.title('MNIST Handwritten Digit Generator')
st.write('Select a digit (0-9) and generate 5 handwritten-style images using a trained CVAE.')

digit = st.selectbox('Choose a digit to generate:', list(range(10)), index=0)

if st.button('Generate Images'):
    device = 'cpu'
    model = load_model(device=device)
    images = generate_images(model, digit, num_images=5, device=device)
    show_images(images) 