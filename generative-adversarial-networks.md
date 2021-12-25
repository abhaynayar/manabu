# Generative adversarial networks
## Generative models

- Variational Autoencoders: Encoder -> Latent space -> Decoder
- Generative adversarial networks: Noise -> Generator -> Discriminator

## Binary Cross Entropy - Cost Function

```
# m: batch size
# h: predictions
# y: labels
# x: features
# theta: parameters

for i = 1 to m:
    J += y[i] * log(h(x[i], theta))          # relevant when label is 1
    J += (1-y[i]) * log(1 - h(x[i], theta))  # relevant when label is 0


# Average loss over batch
J *= -1/m
```

## Training GANs

Discriminator:

```
Noise
-> Generator
-> Fakes
-> {Reals, Fakes}
-> Discriminator
-> Cost
-> Update discriminator paramters
```

Generator:

```
Noise
-> Generator
-> Fakes *only*
-> Discriminator
-> Cost
-> Update generator parameters
```


Both should be at similar skill level.


## Intro to pytorch


```py
import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, in):
        super().__init__()
        self.log_reg = nn.Sequential(
        nn.Linear(in, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        return self.log_reg(x)


model = LogisticRegression(16)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for t in range(n_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


A tensor is a generalization of higher-ranked matrices.

```py
import torch

example_tensor = torch.Tensor(
    [
     [[1, 2], [3, 4]],
     [[5, 6], [7, 8]],
     [[9, 0], [1, 2]]
    ]
)

print(example_tensor.device)
print(example_tensor.shape)
print("shape[0] =", example_tensor.shape[0])
print("size(1) =", example_tensor.size(1))
print("Rank =", len(example_tensor.shape))
print("Number of elements =", example_tensor.numel())
example_scalar = example_tensor[1, 1, 0]
example_scalar.item()
print(example_tensor[:, 0, 0])
```

Initializing tensors:

```py
torch.ones_like(example_tensor)
torch.zeros_like(example_tensor)
torch.randn_like(example_tensor)
torch.randn(2, 2, device='cpu')
```

Basic functions:

```py
print("Mean:", example_tensor.mean())
print("Stdev:", example_tensor.std())
```

Neural network module:

```py
import torch.nn as nn

linear = nn.Linear(10, 2)
example_input = torch.randn(3, 10)
example_output = linear(example_input)

relu = nn.ReLU()
relu_output = relu(example_output)

batchnorm = nn.BatchNorm1d(2)
batchnorm_output = batchnorm(relu_output)

mlp_layer = nn.Sequential(
    nn.Linear(5, 2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)

test_example = torch.randn(5,5) + 1
print("input: ")
print(test_example)
print("output: ")
print(mlp_layer(test_example))
```

Optimizers:

```
import torch.optim as optim
adam_opt = optim.Adam(mlp_layer.parameters(), lr=1e-1)
```

Training loop:

```
train_example = torch.randn(100,5) + 1
adam_opt.zero_grad()

cur_loss = torch.abs(1 - mlp_layer(train_example)).mean()

cur_loss.backward()
adam_opt.step()
print(cur_loss)
```


## Inputs to pre-trained GAN

- Truncate the normal distribution (from where you get z) to tune
  quality versus diversity of the generated features.


## GAN to generate handwritten digits

```
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.gen(noise)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
```

