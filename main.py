import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import string
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# nltk.download('punkt_tab')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load dataset
ds = load_dataset("daspartho/mrbeast-thumbnails")
data = ds['train']

images = [img for img in data['image']]
titles = [t for t in data['title']]

transform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.CenterCrop([64,64]),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)])


img_tensor = [transform(img) for img in images]

img_batch = torch.stack(img_tensor)


# Text preprocessing: Tokenize and clean
tokenized_titles = []
all_tokens = []

for title in titles:
    tokens = nltk.word_tokenize(title.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokenized_titles.append(tokens)
    all_tokens.extend(tokens)

# Build vocabulary
vocab = list(set(all_tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Convert each title to BoW vector
bow_vectors = []

for tokens in tokenized_titles:
    bow = torch.zeros(len(vocab))
    for token in tokens:
        if token in word2idx:
            bow[word2idx[token]] += 1
    bow_vectors.append(bow)

bow_tensor = torch.stack(bow_vectors)

print(f"BoW tensor shape: {bow_tensor.shape}") 

class MrBeastDataset(Dataset):
    def __init__(self, image_tensors, bow_tensors):
        self.images = image_tensors
        self.bows = bow_tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.bows[idx], self.images[idx]
    
# Create the dataset
dataset = MrBeastDataset(img_tensor, bow_vectors)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(12288 + vocab_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, images, bows):
        x = images.view(images.size(0), -1)  # Flatten image to [batch, 12288]
        x = torch.cat([x, bows], dim=1)      # Concatenate image and title (BoW)
        return self.model(x)

  
class Generator(nn.Module):
    def __init__(self, vocab_size, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + vocab_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, 12288),  # Output size for 3x64x64
            nn.Tanh()
        )

    def forward(self, noise, bows):
        x = torch.cat([noise, bows], dim=1)  # Combine noise and title BoW
        x = self.model(x)
        return x.view(x.size(0), 3, 64, 64)  # Reshape to image

vocab_size = len(vocab)
noise_dim = 100
discriminator = Discriminator(vocab_size).to(device)
generator = Generator(vocab_size).to(device)

vocab_size = len(vocab)
noise_dim = 100
discriminator = Discriminator(vocab_size).to(device)
generator = Generator(vocab_size).to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, vocab_size):
    g_optimizer.zero_grad()

    # Sample random noise
    z = torch.randn(batch_size, 100).to(device)

    # Randomly sample conditional text (BoW) from existing vocab
    # For now, use random real BoW samples — better than making random ones
    random_indices = np.random.randint(0, len(bow_tensor), batch_size)
    conditional_bow = bow_tensor[random_indices].to(device)

    # Generate fake images
    fake_images = generator(z, conditional_bow)

    # Try to fool the discriminator
    labels_real = torch.ones(batch_size, 1).to(device)
    validity = discriminator(fake_images, conditional_bow)

    g_loss = criterion(validity, labels_real)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()
def discriminator_train_step(real_images, bows, discriminator, generator, d_optimizer, criterion):
    batch_size = real_images.size(0)
    d_optimizer.zero_grad()

    # === Real Images ===
    real_labels = torch.ones(batch_size, 1).to(device)
    output_real = discriminator(real_images.to(device), bows.to(device))
    d_loss_real = criterion(output_real, real_labels)

    # === Fake Images ===
    z = torch.randn(batch_size, 100).to(device)
    fake_images = generator(z, bows.to(device))  # use same BoW
    fake_labels = torch.zeros(batch_size, 1).to(device)
    output_fake = discriminator(fake_images.detach(), bows.to(device))  # detach so G doesn't get gradients
    d_loss_fake = criterion(output_fake, fake_labels)

    # === Total Loss ===
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()
epochs = 30

# import matplotlib.pyplot as plt
def show_fake_thumbnail(generator, bow_vector):
    z = torch.randn(1, 100).to(device)
    bow_vector = bow_vector.unsqueeze(0).to(device)
    fake_img = generator(z, bow_vector).detach().cpu()
    fake_img = fake_img.squeeze().permute(1, 2, 0) * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
    plt.imshow(fake_img.numpy())
    plt.axis("off")
    plt.show()


for epoch in range(epochs):
    g_running_loss = 0.0
    d_running_loss = 0.0

    for bows, real_images in dataloader:
        # === Train Discriminator ===
        d_loss = discriminator_train_step(real_images, bows, discriminator, generator, d_optimizer, criterion)
        
        # === Train Generator ===
        g_loss = generator_train_step(real_images.size(0), discriminator, generator, g_optimizer, criterion, vocab_size)

        d_running_loss += d_loss
        g_running_loss += g_loss

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_running_loss:.4f} | G Loss: {g_running_loss:.4f}")
        if epoch % 5 == 0:
          show_fake_thumbnail(generator, bow_tensor[0])

def title_to_bow(title, word2idx, vocab_size):
    tokens = nltk.word_tokenize(title.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    
    bow = torch.zeros(vocab_size)
    for token in tokens:
        if token in word2idx:
            bow[word2idx[token]] += 1
    return bow
def generate_thumbnail_from_title(title, generator, word2idx, vocab_size):
    bow_vector = title_to_bow(title, word2idx, vocab_size).unsqueeze(0).to(device)
    z = torch.randn(1, 100).to(device)
    with torch.no_grad():
        fake_image = generator(z, bow_vector).cpu().squeeze()
    
    # Denormalize image (since Tanh outputs in [-1, 1])
    fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)
    img = fake_image.permute(1, 2, 0).numpy()
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'"{title}"')
    plt.show()
title_input = "I gave $100,000 to a homeless person"
generate_thumbnail_from_title(title_input, generator, word2idx, vocab_size)
         
