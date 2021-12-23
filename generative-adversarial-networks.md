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
