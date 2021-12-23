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
