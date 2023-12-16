---
layout: post
title: "MLX: Apple silicon 용 Machine Learning 프레임워크 - 02.Regression example"
tags: archive
---

- [MLX examples](#mlx-examples)
  - [Linear Regression](#linear-regression)
    - [CPU 연산과의 수행시간 비교](#cpu-연산과의-수행시간-비교)
  - [Logistic regression](#logistic-regression)


# MLX examples

몇 가지 Machine learning 예제들을 통해 MLX 사용법을 익혀보도록 하겠습니다. 사실 numpy 나 torch, scipy 등의 라이브러리에 익숙하다면 그렇게 크게 어렵지 않기 때문에 잘 따라할 수 있습니다. 메쏘드들이 거의 유사하게 구현되었거든요.

## Linear Regression

아주 간단한 Linear Regression 예제를 돌려보겠습니다. 임의의 함수를 만들어 데이터를 합성하고, 해당 데이터를 이용해 역으로 함수를 approximate 하는 예제입니다.  

우선 관련 모듈들을 import 하고 hyperparam 들을 세팅해줍니다.


```python
import mlx.core as mx
import time

num_features = 100
num_examples = 1_000
test_examples = 100
num_iters = 10_000  # iterations of SGD
lr = 0.01  # learning rate for SGD
```

임의의 선형 함수를 만들어주고, 임의의 input 데이터를 만들어줍니다. 모두 `mx.random.normal` 을 이용해 랜덤하게 만들어 주겠습니다.  
label 값의 경우 만들어진 input 데이터를 함수에 통과시키고, 작은 noise 를 부여하여 만들어줍니다.


```python
# 임의의 선형 함수 True parameters
w_star = mx.random.normal((num_features,))

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps
```

별도의 테스트셋도 만들어주겠습니다.


```python
# Test set generation
X_test = mx.random.normal((test_examples, num_features))
y_test = X_test @ w_star
```

이제 Loss function 과 Gradient function 을 만들어줍니다. 우선은 MSE loss 를 사용합니다.  
$${L}_{MSE}=\frac{1}{2n}\sum{(y-pred)^2}$$
Gradient function 의 경우 수식을 구현해서 사용해도 좋지만, `mx.grad` 를 이용하면 loss function 으로부터 바로 얻어낼 수 있습니다. 
$$\nabla{L}_{MSE}=\frac{1}{n}X^T(Xw-y)$$


```python
# MSE Loss function
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

# Gradient function
grad_fn = mx.grad(loss_fn)

# 실제 수학적 구현은 아래와 같다. 수행시간도 동일함
# def grad_fn(w):
#     return X.T @ (X @ w-y) * (1/num_examples)
```

이제 Linear regression 을 위한 parameter 를 초기화하고 SGD (Stochastic Gradient Descent) 방법을 이용해 학습합니다.  
Test set 에 대해서 MSE 값이 크게 감소한 것을 확인할 수 있습니다. 또 수행시간은 약 0.9초 남짓하게 걸리네요. (M1 맥북 pro 기준입니다) **Throughput 은 약 10K iter/s 입니다.**


```python
# Initialize random parameter
w = 1e-2 * mx.random.normal((num_features,))

# Test error (MSE)
pred_test = X_test @ w
test_error = mx.mean(mx.square(y_test - pred_test))
print(f"Initial test error (MSE): {test_error.item():.5f}")
```

    Initial test error (MSE): 90.18447



```python
# Training by SGD
start = time.time()
for its in range(1,num_iters+1):
    grad = grad_fn(w)
    w = w - lr * grad
mx.eval(w)
end = time.time()
print(f"Training elapsed time: {end-start} seconds")
print(f"Throughput {num_iters/(end-start):.3f} it/s")
```

    Training elapsed time: 0.9663820266723633 seconds
    Throughput 10347.875 it/s



```python
# Test error (MSE)
pred_test = X_test @ w
test_error = mx.mean(mx.square(y_test - pred_test))
print(f"Final test error (MSE): {test_error.item():.5f}")
```

    Final test error (MSE): 0.00001


### CPU 연산과의 수행시간 비교

만약 numpy array 를 사용했다면 그 속도는 어떻게 될까요? 동일한 구현 방식으로 연산 수행 속도를 비교해 보겠습니다. 물론 numpy 모듈은 Apple silicon 의 GPU 를 사용할 수 없습니다.


```python
# True parameters
w_star = np.random.normal(size=(num_features,1))

# Input examples (design matrix)
X = np.random.normal(size=(num_examples, num_features))

# Noisy labels
eps = 1e-2 * np.random.normal(size=(num_examples,1))
y = np.matmul(X, w_star) + eps

# Test set generation
X_test = np.random.normal(size=(test_examples, num_features))
y_test = np.matmul(X_test, w_star)
```


```python
def loss_fn(w):
    return 0.5 * np.mean(np.square(np.matmul(X, w) - y))

def grad_fn(w):
    return np.matmul(X.T, np.matmul(X, w)-y) * (1/num_examples)
```


```python
w = 1e-2 * np.random.normal(size=(num_features,1))

pred_test = np.matmul(X_test, w)
test_error = np.mean(np.square(y_test - pred_test))
print(f"Initial test error (MSE): {test_error.item():.5f}")
```

    Initial test error (MSE): 51.48214



```python
start = time.time()
for its in range(1,num_iters+1):
    grad = grad_fn(w)
    w = w - lr * grad
    
end = time.time()
print(f"Training elapsed time: {end-start} seconds")
print(f"Throughput {num_iters/(end-start):.3f} it/s")
```

    Training elapsed time: 1.2018659114837646 seconds
    Throughput 8320.396 it/s



```python
pred_test = np.matmul(X_test, w)
test_error = np.mean(np.square(y_test - pred_test))
print(f"Final test error (MSE): {test_error.item():.5f}")
```

    Final test error (MSE): 0.00001


수행시간이 약 1.2초로 증가한 것을 확인할 수 있습니다. **Throughput은 8.3K 정도로 약간 감소했네요.** feature 도 100개 정도인 아주 간단한 linear regression 에서도 이 정도의 속도 증가는 기대할 수 있겠습니다. 드라마틱한 변화는 발생하지 않는 걸 볼 수 있습니다.

## Logistic regression

Logistic regression 도 잠깐 테스트를 해보도록 하겠습니다. 모든 과정은 Linear regression 과 동일하지만 Label 이 0,1 로 구성된다는 것, activation function 을 사용한다는 것, Loss 로 Cross entropy 를 사용한다는 것만 다릅니다.


```python
import time

import mlx.core as mx

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01

# True parameters
w_star = mx.random.normal((num_features,1))

# Input examples
X = mx.random.normal((num_examples, num_features))

# Labels (with noise)
eps = 1e-2 * mx.random.normal((num_examples,1))
y_bool = (X @ w_star + eps) > 0
y = mx.where(y_bool, 1.0, 0.0)

# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,1))

def loss_fn(w):
    logits = X @ w
    y_hat = mx.sigmoid(logits)
    return mx.mean(-y * mx.log(y_hat) - (1-y) * mx.log(1-y_hat))

# def grad_fn(w):
#     logits = X @ w
#     y_hat = mx.sigmoid(logits)
#     return X.T @ (y_hat - y) * (1/num_examples)
    
grad_fn = mx.grad(loss_fn)

tic = time.time()
for iters in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad

    # Early stopping
    preds = (X @ w) > 0
    acc = mx.mean(preds == y_bool)
    if acc > 0.95:
        print(f"Early stop at iteration {iters}")
        break

mx.eval(w)
toc = time.time()

loss = loss_fn(w)
final_preds = (X @ w) > 0
acc = mx.mean(final_preds == y)

throughput = iters / (toc - tic)
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s)"
)
```

    Early stop at iteration 256
    Loss 0.44712, Accuracy 0.95000 Throughput 1627.90763 (it/s)



```python
import time

import numpy as np

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01

# True parameters
w_star = np.random.normal(size=(num_features,))

# Input examples
X = np.random.normal(size=(num_examples, num_features))

# Labels
eps = 1e-2 * np.random.normal(size=(num_examples,))
y_bool = np.matmul(X, w_star) + eps > 0
y = np.where(y_bool, 1.0, 0.0)


# Initialize random parameters
w = 1e-2 * np.random.normal(size=(num_features,))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_fn(w):
    logits = np.matmul(X, w)
    y_hat = sigmoid(logits)
    return np.mean(-y * np.log(y_hat) - (1-y) * np.log(1-y_hat))

def grad_fn(w):
    logits = np.matmul(X, w)
    y_hat = sigmoid(logits)
    return np.matmul(X.T, y_hat - y) * (1/num_examples)

tic = time.time()
for iters in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad

    # Early stopping
    preds = np.matmul(X, w) > 0
    acc = np.mean(preds == y_bool)
    if acc > 0.95:
        print(f"Early stop at iteration {iters}")
        break

toc = time.time()

loss = loss_fn(w)
final_preds = np.matmul(X, w) > 0
acc = np.mean(final_preds == y)

throughput = iters / (toc - tic)
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s)"
)
```

    Early stop at iteration 297
    Loss 0.40303, Accuracy 0.95100 Throughput 3858.49777 (it/s)


재미있게도 numpy 를 사용하는 것이 throughput 이 더 이득이네요.  
추후 multi-layer perceptron 처럼 행렬 연산이 더 무거워지는 경우 어떻게 되는지 확인해보도록 하겠습니다.
