# Neural networks and deep learning
## Logistic regression

Linear regression:

```
z = w.T*x + b
z = w1*x1 + w2*y2 + b
```

Logistic regression:

```
a = sigmoid(w.T*x + b)   # Activation function
a = sigmoid(z)   # sigmoid(z) = 1 / (1+e**-z)
```

## Cost function

Single training example: loss function.

```
L(a,y) = (1/2)*(a-y)**2   # Not used since it is not convex
L(a,y) = -(y*log(a) + (1-y)*log(1-a))
```

Entire training set: cost function.

```
J(w,b) = (1/m)*sum(L(a[i],y))
```

## Gradient descent

```
repeat {
  w := w - alpha*[dJ(w,b)/dw]
  b := b - alpha*[dJ(w,b)/db]
}

# alpha = learning rate
```

## Logistic regression derivatives

```
da = -(y/a) + [(1-y)/(1-a)]
dz = a - y   # https://community.deeplearning.ai/t/derivation-of-dl-dz/165
dw1 = x1*dz
dw2 = x2*dz
db = dz
```

## Logistic regression on m examples

```
J = 0
dw = np.zeros((n-x,1))
db = 0

for i=1 to m:
    z[i] = w.T*x[i] + b
    a[i] = sigmoid(z[i])
    J += -[y[i]*log(a[i]) + (1-y[i])*log(1-a[i])]
    dz[i] = a[i] - y[i]
    dw += x[i]*dz[i]

J /= m
dw /= m
db /= m

# Gradient descent
w = w - alpha*dw
b = b - alpha*db
```

## Vectorizing logistic regression

Over m training examples:

```
Z = np.dot(w.T,X) + b
A = sigmoid(Z)
```

Gradient output:

```
dZ = A - Y
dw = (1/m) * X * dZ.T
db = (1/m) * np.sum(dZ)
```


## Building basic functions with numpy


```py
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2],1))
    return v

def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    return s

def L1(yhat, y):
    loss = np.sum(abs(y - yhat))
    return loss

def L2(yhat, y):
    loss = np.sum(np.dot(y-yhat, y-yhat))
    return loss
```


## Logistic Regression - Code


```py
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m) * ((np.dot(np.log(A), Y.T)) + np.dot(np.log(1-A), (1-Y).T))
    
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
```

Cost curve:

```py
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()
```

Learning rates:

```py   
learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

Test an image:

```py
my_image = "my_image.jpg"
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"],
    logistic_regression_model["b"], image)
```

## Neural network representation

First neuron of first hidden layer:

```
z1[1] = np.dot(w1[1].T, x) + b1[1]
```

The entire first layer:

```
Z1 = np.dot(W1.T, X) + b1
```

Across multiple examples:

```
for i=1 to m:
    z[1](i) = np.dot(w[1], x[1] + b[1])
    a[1](i) = sigmoid(z[1](i))
    z[2](i) = np.dot(w[2], a[1](i) + b[2])
    a[2](i) = sigmoid(z[2](i)
```

Vectorized across multiple examples:

```
Z[1] = np.dot(W[1],X) + b[1]
A[1] = sigmoid(Z[1])
Z[2] = np.dot(W[2],A[1]) + b[2]
A[2] = sigmoid(Z[2])
```

## Activation functions

- Sigmoid is preferred for the output layer in binary classification.
  (since it outputs between zero and one)
- tanh: `a = (e**z - e**-z) / (e**z + e**-z)`
- ReLU: `a = max(0,z)`
- Leaky ReLU: `a = max(0.01*z, z)`


## Why do we use non-linear activation functions?

Linear activation functions: useless for hidden layers as the composition
of two linear functions is still linear. So it becomes no different than
the one neuron linear regression.

You can still use a linear activation function in the output layer. But use
non-linearities in the hidden layers.


## Derivatives of activation functions

- Sigmoid: `a' = a*(1-a)`
- Tanh: `a' = 1 - a**2`
- ReLU: `g'(z) = {0 if z<0, 1 if z>0, undefine if z=0}`
- Leaky ReLU: `g'(z) = {0.01 if z<0, 1 if z>=0}`

## Gradient descent for neural networks






