---
layout: post
title: "MLX: A Machine Learning Framework for Apple Silicon - 02. Regression example"
tags: archive
lang: en
---

- [MLX examples](#mlx-examples)
  - [Linear Regression](#linear-regression)
    - [Comparing execution time with CPU computation](#cpu-연산과의-수행시간-비교)
  - [Logistic regression](#logistic-regression)


# MLX examples

Let's get familiar with how to use MLX through a few machine learning examples. Actually, if you're familiar with libraries like numpy, torch, or scipy, it isn't all that difficult, so you'll be able to follow along well. The methods are implemented almost identically.

## Linear Regression

Let's run a very simple Linear Regression example. It's an example where we create an arbitrary function to synthesize data, and then use that data to inversely approximate the function.

First, we import the relevant modules and set up the hyperparameters.


```python
import mlx.core as mx
import time

num_features = 100
num_examples = 1_000
test_examples = 100
num_iters = 10_000  # iterations of SGD
lr = 0.01  # learning rate for SGD
```

We create an arbitrary linear function and arbitrary input data. We'll create everything randomly using `mx.random.normal`.
For the label values, we pass the created input data through the function and add a small amount of noise.


```python
# Arbitrary linear function True parameters
w_star = mx.random.normal((num_features,))

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps
```

Let's also create a separate test set.


```python
# Test set generation
X_test = mx.random.normal((test_examples, num_features))
y_test = X_test @ w_star
```

Now we create the loss function and the gradient function. For now, we use MSE loss.
$${L}_{MSE}=\frac{1}{2n}\sum{(y-pred)^2}$$
For the gradient function, you can implement and use the formula directly, but if you use `mx.grad`, you can obtain it directly from the loss function.
$$\nabla{L}_{MSE}=\frac{1}{n}X^T(Xw-y)$$


```python
# MSE Loss function
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

# Gradient function
grad_fn = mx.grad(loss_fn)

# The actual mathematical implementation is as follows. The execution time is the same.
# def grad_fn(w):
#     return X.T @ (X @ w-y) * (1/num_examples)
```

Now we initialize the parameters for linear regression and train using the SGD (Stochastic Gradient Descent) method.
We can confirm that the MSE value on the test set decreased significantly. Also, the execution time takes about 0.9 seconds. (Based on the M1 MacBook Pro) **The throughput is about 10K iter/s.**


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


### Comparing execution time with CPU computation

What would the speed be if we used numpy arrays? Let's compare the computation execution speed using the same implementation approach. Of course, the numpy module cannot use the Apple silicon GPU.


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


We can confirm that the execution time increased to about 1.2 seconds. **The throughput decreased slightly to about 8.3K.** Even in a very simple linear regression with around 100 features, we can expect this much of a speed increase. We can see that no dramatic change occurs.

## Logistic regression

Let's also briefly test logistic regression. The entire process is the same as linear regression, except that the labels consist of 0 and 1, that an activation function is used, and that cross entropy is used as the loss.


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


Interestingly, using numpy gives a better throughput.
Later, I'll check what happens when the matrix operations become heavier, such as in a multi-layer perceptron.
