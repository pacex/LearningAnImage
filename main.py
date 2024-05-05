import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Use CUDA if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print("Using device: " + device)

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for layer in self.hidden_layers:
            out = self.relu(layer(out))
        out = self.output_layer(out)
        return out

# Frequency Encoding
class FrequencyEncoding(nn.Module):
    def __init__(self, input_dim, n_frequencies):
        super(FrequencyEncoding, self).__init__()
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.output_dim = input_dim * n_frequencies * 2

        self.frequencies = torch.tensor([pow(2,x) * np.pi for x in range(n_frequencies)], device=device).repeat(input_dim)
    
    def forward(self, x):
        expanded_x = x.unsqueeze(2).repeat(1, 1, self.n_frequencies).view(x.size(0), -1)

        enc_sin = torch.sin(expanded_x * self.frequencies.unsqueeze(0))
        enc_cos = torch.cos(expanded_x * self.frequencies.unsqueeze(0))
        return torch.cat((enc_sin, enc_cos), dim=1)
    
# MLP with Encoding
class NetworkWithEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers, n_frequencies):
        super(NetworkWithEncoding, self).__init__()
        self.encoding = FrequencyEncoding(input_dim=input_dim, n_frequencies=n_frequencies)
        self.mlp = MLP(input_dim=self.encoding.output_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_hidden_layers=n_hidden_layers)

    def forward(self, x):
        e = self.encoding(x)
        y = self.mlp(e)
        return y


class ImagePixelDataset(Dataset):
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size

    def __len__(self):
        return self.width * self.height

    def __getitem__(self, idx):
        y, x = divmod(idx, self.width)  # Calculate pixel coordinates
        r, g, b = self.image.getpixel((x, y))  # Get RGB values of the pixel
        # Normalize x, y to range [0, 1]
        x_norm = x / (self.width - 1)
        y_norm = y / (self.height - 1)
        # Normalize r, g, b to range [0, 1]
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        return torch.tensor([x_norm, y_norm]), torch.tensor([r_norm, g_norm, b_norm])

def saveMLPImage(model, width, height, fname):
    # Create a grid of pixel positions
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    positions = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    positions_tensor = torch.tensor(positions, dtype=torch.float32)

    # Move positions tensor to the appropriate device (CPU or GPU)
    positions_tensor = positions_tensor.to(device)

    # Pass the positions tensor through the model to get predicted RGB values
    model.eval()
    with torch.no_grad():
        predicted_rgb = model(positions_tensor)

    # Reshape the predicted RGB values into an image format
    predicted_rgb_image = predicted_rgb.cpu().numpy().reshape((height, width, 3))
    predicted_rgb_image = np.clip(predicted_rgb_image, 0, 1)

    # Save the generated image to a file
    plt.imsave(fname, predicted_rgb_image)


# Initialize the model
model = NetworkWithEncoding(input_dim=2, hidden_dim=256, output_dim=3, n_hidden_layers=4, n_frequencies=32)
model.to(device)

# Load data
print("Loading image data...")
image_path = "house.jpg"
batch_size = 64
image_dataset = ImagePixelDataset(image_path)
train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

# Train the model
print("Training the model...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + "...")
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 512 == 0: 
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), running_loss / 100))
            running_loss = 0.0
    print("Writing image file...")
    saveMLPImage(width=image_dataset.width, height=image_dataset.height, model=model, fname="images/ep"+str(epoch+1)+"of"+str(num_epochs)+".png")
    print("Done.")

print("Training Done.")