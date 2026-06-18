---
layout: post
title: "MLX: A Machine Learning Framework for Apple Silicon - 03. Multi-Layer Perceptron example"
tags: archive
lang: en
---


# Multi-Layer Perceptron (MLP) example

Let's implement a Multi-Layer Perceptron (MLP) example using the MNIST dataset, with MLX.
We're going to compare how it differs from implementing it with Torch.

## Implementing an MLP with MLX

First, let's import the relevant modules.


```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np
from matplotlib import pyplot as plt

from time import time
```

Let's create an MLP class. If you look closely, you can see it's almost identical to using torch.nn.
I suspect they considered things like this while building the library. Wasn't the intention that, without touching the code, just changing the import part from `import torch.nn as nn` to `import mlx.nn as nn` and so on would let the code run? (just my speculation)


```python
class MLP(nn.Module):
    def __init__(
        self, 
        num_layers: int,
        input_dims: int, 
        hidden_dims: int,
        output_dims: int
    ):
        super().__init__()
        layer_sizes = [input_dims] + [hidden_dims] * num_layers + [output_dims]
        self.layers = self._make_layers(layer_sizes)
    
    def _make_layers(self, layer_sizes):
        layers = []
        for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [
                            nn.Linear(idim, odim), 
                            nn.ReLU()
                       ]
        
        return nn.Sequential(*layers[:-1])
    
    def __call__(self, x):
        return self.layers(x)
```

Let's also create a loss function and an evaluation function.
These too are implemented almost identically to torch.


```python
def loss_fn(model, X, y):
    # nn.losses.cross_entropy computes the loss between the logits and the target
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)
```

Let's set the hyperparameters, download the MNIST dataset, and preprocess it.
Since we're using an MLP, we flatten the ( 28 X 28 ) images into 768 dimensions.


```python
num_layers = 2
hidden_dim = 256
num_classes = 10
batch_size = 256
num_epochs = 10
learning_rate = 1e-2

# Load the data
import mnist
train_images, train_labels, test_images, test_labels = map(
    mx.array, [
        mnist.train_images(),
        mnist.train_labels(),
        mnist.test_images(),
        mnist.test_labels(),
    ]
)
# Flatten the images
train_images = mx.reshape(train_images, [train_images.shape[0],-1])
valid_images, test_images = test_images[:-10], test_images[-10:]
valid_labels, test_labels = test_labels[:-10], test_labels[-10:]
valid_images = mx.reshape(valid_images, [valid_images.shape[0],-1])
```

Let's implement a batch iterator. torch has a dataloader implemented. As for MLX, I haven't yet found one implemented separately. (as of 2023/12/17)


```python
def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]
```

Now let's train the MLP we made.
The part that differs from PyTorch is the update part.
In the case of PyTorch, after defining only the loss function, you do the update via `loss.backward()`, `optimizer.step()`, and `optimizer.zero_grad()`.
MLX computes the loss and gradient via `nn.value_and_grad`, and updates the model via `optimizer.update(model, gradient)`.

For now, let's train simply with SGD without using anything like a learning scheduler. Since parameter initialization is also random, sometimes training doesn't work, so be careful!!


```python
# Load the model
model = MLP(num_layers=num_layers, 
            input_dims=train_images.shape[-1],
            hidden_dims=hidden_dim,
            output_dims=num_classes)
mx.eval(model.parameters())

# loss and grad fn
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# optimizer
optimizer = optim.SGD(learning_rate=learning_rate)

accuracy = []
tic = time()
for epoch in range(num_epochs):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        
        # is this really necessary?    
    accuracy += [eval_fn(model, valid_images, valid_labels).item()]

mx.eval(model.parameters(), optimizer.state)
toc = time()

print(f"Training time: {(toc-tic)/num_epochs:.2f} sec/epoch")
   
plt.figure(figsize=(4,3))
plt.plot(range(1,num_epochs+1), accuracy)
plt.plot(range(1,num_epochs+1),[1.0]*num_epochs, ls='--')
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()

```

    Training time: 0.21 sec/epoch

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/5e4d392e-354b-42b5-98dc-cf700e201a4f)
    


As you can see, it takes about 0.2 seconds per epoch. Running it several times, the accuracy is sometimes 0.2 and sometimes rises up to 0.9. Let's aim for a time when it trains well and evaluate on the test set.


```python
num_images = len(test_images)

# Set the number of images to display per row. This value can be adjusted as needed.
images_per_row = 5

# Calculate the total number of rows.
num_rows = (num_images + images_per_row - 1) // images_per_row

# Create subplots for all rows and columns.
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))

# Display the image, prediction, and ground truth on each subplot.
for i, (test_img, test_lb) in enumerate(zip(test_images, test_labels)):
    row = i // images_per_row
    col = i % images_per_row
    ax = axes[row, col]
    
    pred = mx.argmax(model(test_img.reshape([1,-1])), axis=1).item()
    ax.imshow(np.array(test_img.reshape(28, 28) * 255), cmap='gray')
    ax.set_title(f'Predict: {pred}\nTrue: {test_lb.item()}')
    ax.axis('off')  # Hide the axes.

# Hide the remaining empty subplots.
for i in range(num_images, num_rows * images_per_row):
    axes[i // images_per_row, i % images_per_row].axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/f5246db2-f772-459d-a091-812d366ba92e)


## Implementing an MLP with PyTorch

Now let's implement the same code with Torch. PyTorch can also use the GPU by using device="mps".
So what is MLX's advantage? It's the unified memory of CPU and GPU. That is, since the memory is shared, the time to move memory to the GPU is reduced.

### PyTorch + CPU training


```python
import torch
import mnist

# Load the data
train_images, train_labels, test_images, test_labels = map(
    torch.Tensor, [
        mnist.train_images(),
        mnist.train_labels(),
        mnist.test_images(),
        mnist.test_labels(),
    ]
)
# Flatten the images
train_labels, test_labels = train_labels.long(), test_labels.long()
train_images = torch.reshape(train_images, [train_images.shape[0],-1])
valid_images, test_images = test_images[:-10], test_images[-10:]
valid_labels, test_labels = test_labels[:-10], test_labels[-10:]
valid_images = torch.reshape(valid_images, [valid_images.shape[0],-1])

class torchMLP(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int,
        input_dims: int, 
        hidden_dims: int,
        output_dims: int
    ):
        super().__init__()
        layer_sizes = [input_dims] + [hidden_dims] * num_layers + [output_dims]
        self.layers = self._make_layers(layer_sizes)
    
    def _make_layers(self, layer_sizes):
        layers = []
        for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [
                            torch.nn.Linear(idim, odim), 
                            torch.nn.ReLU()
                       ]
        
        return torch.nn.Sequential(*layers[:-1])
    
    def __call__(self, x):
        return self.layers(x)

def loss_fn(model, X, y):
    # nn.losses.cross_entropy computes the loss between the logits and the target
    return torch.nn.CrossEntropyLoss()(model(X), y)

def eval_fn(model, X, y):
    return torch.mean((torch.argmax(model(X), axis=1) == y).float())

def batch_iterate(batch_size, X, y):
    perm = torch.randperm(y.size(0))
    for s in range(0, y.size(0), batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

# Load the model
model = torchMLP(num_layers=num_layers, 
            input_dims=train_images.shape[-1],
            hidden_dims=hidden_dim,
            output_dims=num_classes)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

accuracy = [eval_fn(model, valid_images, valid_labels).item()]
tic = time()
for epoch in range(num_epochs):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss = loss_fn(model, X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    accuracy += [eval_fn(model, valid_images, valid_labels).item()]

toc = time()
print(f"Training time: {(toc-tic)/num_epochs:.2f} sec/epoch")

plt.figure(figsize=(4,3))
plt.plot(range(num_epochs+1), accuracy)
plt.plot(range(num_epochs+1),[1.0]*(num_epochs+1), ls='--')
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()
```

    Training time: 0.39 sec/epoch
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/4a198f2c-b5af-42cf-af64-16b6c23394e8)


### PyTorch + GPU (mps) training


```python
import torch
import mnist

device = torch.device("mps:0") if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

# Load the data
train_images, train_labels, test_images, test_labels = map(
    torch.Tensor, [
        mnist.train_images(),
        mnist.train_labels(),
        mnist.test_images(),
        mnist.test_labels(),
    ]
)
# Flatten the images
train_labels, test_labels = train_labels.long(), test_labels.long()
train_images = torch.reshape(train_images, [train_images.shape[0],-1])
valid_images, test_images = test_images[:-10], test_images[-10:]
valid_labels, test_labels = test_labels[:-10], test_labels[-10:]
valid_images = torch.reshape(valid_images, [valid_images.shape[0],-1])

class torchMLP(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int,
        input_dims: int, 
        hidden_dims: int,
        output_dims: int
    ):
        super().__init__()
        layer_sizes = [input_dims] + [hidden_dims] * num_layers + [output_dims]
        self.layers = self._make_layers(layer_sizes)
    
    def _make_layers(self, layer_sizes):
        layers = []
        for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [
                            torch.nn.Linear(idim, odim), 
                            torch.nn.ReLU()
                       ]
        
        return torch.nn.Sequential(*layers[:-1])
    
    def __call__(self, x):
        return self.layers(x)

def loss_fn(model, X, y):
    # nn.losses.cross_entropy computes the loss between the logits and the target
    return torch.nn.CrossEntropyLoss()(model(X), y)

def eval_fn(model, X, y):
    return torch.mean((torch.argmax(model(X), axis=1) == y).float())

def batch_iterate(batch_size, X, y):
    perm = torch.randperm(y.size(0))
    for s in range(0, y.size(0), batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

# Load the model
model = torchMLP(num_layers=num_layers, 
            input_dims=train_images.shape[-1],
            hidden_dims=hidden_dim,
            output_dims=num_classes)
model.to(device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

accuracy = [eval_fn(model, valid_images.to(device), valid_labels.to(device)).item()]
tic = time()
for epoch in range(num_epochs):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model, X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    accuracy += [eval_fn(model, valid_images.to(device), valid_labels.to(device)).item()]

toc = time()
print(f"Training time: {(toc-tic)/num_epochs:.2f} sec/epoch")

plt.figure(figsize=(4,3))
plt.plot(range(num_epochs+1), accuracy)
plt.plot(range(num_epochs+1),[1.0]*(num_epochs+1), ls='--')
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()
```

    Device: mps:0
    Training time: 0.53 sec/epoch
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/4e500180-e832-407d-a763-edde627bcb3d)


Actually, in the case of an MLP, GPU utilization is lower than for structures like a CNN. That means it's hard to see a big time-reduction benefit from using the GPU.
In fact, looking at the results of the code implemented with PyTorch, training using the CPU took **0.39 seconds per epoch**, but training using the GPU took **0.53 seconds per epoch**. Since it's not Unified Memory, there's a time loss from moving memory to the device, and we could judge that it's also less optimized.

## Conclusion

We implemented a very simple MLP using MLX. We could confirm that using MLX increases the training speed when the model size grows larger than the previous Linear regression. Next time, let's find out how much throughput we can get for a larger model, like a Transformer-structured LLM.

## References

[https://ml-explore.github.io/mlx/build/html/examples/mlp.html](https://ml-explore.github.io/mlx/build/html/examples/mlp.html)


