---
layout: post
title: "MLX: Apple silicon 용 Machine Learning 프레임워크 - 03.Multi-Layer Perceptron example"
tags: archive
---


# Multi-Layer Perceptron (MLP) example

MNIST 데이터셋을 이용한 Multi-Layer Perceptron (MLP) 예제를, MLX 를 이용해 구현해보도록 하겠습니다.
Torch 로 구현하는 것과 어떤 차이를 보이는지 비교해볼 예정입니다.

## MLX 를 이용한 MLP 구현

우선 관련 모듈들을 import 해주겠습니다.


```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np
from matplotlib import pyplot as plt

from time import time
```

MLP class 를 하나 만들어줍니다. 잘 보면 torch.nn 을 사용할 때와 거의 유사함을 확인할 수 있습니다.  
라이브러리를 만들면서 이런 점들을 고려하지 않았을까 싶네요. 코드들을 건드리지 않고 import 하는 부분만 `import torch.nn as nn` 에서 `import mlx.nn as nn` 등으로 바꾸기만 해도 코드가 돌아갈 수 있는 것을 의도하지 않았을까요 (뇌피셜)  


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

Loss function 과 evaluation function 도 만들어주겠습니다.
이 또한 torch 와 거의 유사하게 구현되었습니다.


```python
def loss_fn(model, X, y):
    # nn.losses.cross_entropy 는 logit 과 target 사이의 loss 를 계산해줌
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)
```

Hyperparam 들을 설정하고, MNIST 데이터셋을 다운받아 전처리 해주도록 하겠습니다.  
MLP 를 사용하기 때문에 ( 28 X 28 ) 의 이미지를 768 dimensions 으로 flatten 해줍니다.


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

Batch iterator 를 구현해주겠습니다. torch 는 dataloader 가 구현되어 있죠. 아직 MLX 는 따로 구현되어 있는 것은 찾지 못했습니다. (2023/12/17 기준)


```python
def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]
```

그럼 만들어진 MLP 를 학습시켜 보겠습니다.  
PyTorch 와 다른 부분은 update 부분이 되겠습니다.  
PyTorch 같은 경우 loss function 만 정의한 뒤 `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()` 를 통해 업데이트를 하죠.  
MLX 는 `nn.value_and_grad` 를 통해 loss와 gradient를 구해주고,  `optimizer.update(model, gradient)` 를 통해 model 을 업데이트합니다.  

일단 제일 Learning scheduleer 같은 걸 쓰지 않고 SGD 로 간단하게 학습해보겠습니다. Parameter initialize 도 랜덤이므로 가끔 학습이 안될때도 있으니 주의!!


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
        
        # 이거 꼭 필요할까?    
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
    


보면 1 epoch 당 0.2 초 정도 걸리는 것을 확인할 수 있습니다. 여러번 돌려보니 Accuracy 는 0.2 일때도 있고 0.9까지 오를 때도 있네요. 잘 학습될 때를 노려 test set 평가를 해보겠습니다.


```python
num_images = len(test_images)

# 한 줄에 표시할 그림의 개수를 정합니다. 이 값은 필요에 따라 조정할 수 있습니다.
images_per_row = 5

# 전체 행의 개수를 계산합니다.
num_rows = (num_images + images_per_row - 1) // images_per_row

# 전체 행과 열에 대한 subplot을 생성합니다.
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))

# 각 subplot에 이미지와 예측값, 정답을 표시합니다.
for i, (test_img, test_lb) in enumerate(zip(test_images, test_labels)):
    row = i // images_per_row
    col = i % images_per_row
    ax = axes[row, col]
    
    pred = mx.argmax(model(test_img.reshape([1,-1])), axis=1).item()
    ax.imshow(np.array(test_img.reshape(28, 28) * 255), cmap='gray')
    ax.set_title(f'Predict: {pred}\nTrue: {test_lb.item()}')
    ax.axis('off')  # 축을 숨깁니다.

# 남은 빈 subplot을 숨깁니다.
for i in range(num_images, num_rows * images_per_row):
    axes[i // images_per_row, i % images_per_row].axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/f5246db2-f772-459d-a091-812d366ba92e)


## PyTorch 를 이용한 MLP 구현

이제 동일한 코드를 Torch 로 구현해보도록 하겠습니다. PyTorch 도 device="mps" 를 사용하면 GPU 를 사용할수는 있습니다.
그렇다면 MLX 의 장점은 무엇인가 하면 CPU와 GPU 의 unified memory 라는 점입니다. 즉 메모리를 공유하니까 GPU 로 메모리를 이동시키는 시간이 줄어들겠죠.

### PyTorch + CPU 학습


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
    # nn.losses.cross_entropy 는 logit 과 target 사이의 loss 를 계산해줌
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


### PyTorch + GPU (mps) 학습


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
    # nn.losses.cross_entropy 는 logit 과 target 사이의 loss 를 계산해줌
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


사실 MLP 의 경우 CNN 같은 구조보다 GPU 활용도는 떨어집니다. GPU 를 이용한 시간 단축 효과를 크게 보기 어렵다는 것이죠. 
실제로 PyTorch 로 구현한 코드의 결과를 보면 CPU 를 활용한 학습의 경우 **1 epoch 당 0.39 초**가 걸렸지만, GPU 를 활용한 학습의 경우 **1 epoch 당 0.53 초**가 걸렸습니다. Unified Memory 가 아니기 때문에 메모리를 device 로 옮기는데서 시간 손해를 보기도 하고, 최적화도 덜 되어 있고 판단할 수 있겠습니다.

## Conclusion

아주 간단한 MLP 를 MLX 를 통해 구현해보았습니다. 이전의 Linear regression 보다 모델의 크기가 더욱 커진 상태에서 MLX 를 이용하면 학습 속도가 증가하는 것을 확인할 수 있었습니다. 다음에는 Transformer 구조의 LLM 처럼 더 큰 모델의 경우 어느정도 Throughput 을 낼 수 있을지 알아보겠습니다.

## References

[https://ml-explore.github.io/mlx/build/html/examples/mlp.html](https://ml-explore.github.io/mlx/build/html/examples/mlp.html)


