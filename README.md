# Tiny Word2Vec

This repository is meant as a teaching step to understand how neural networks function. This project is like a tiny word2vec that you can see here:

![word2vec](https://projector.tensorflow.org/)

## Steps that are executed

So as you can see in the code we create a fake vocab list, then we define the model. The `Sequential` class from `pytorch` just enables us to start the pipeline. We first do the `Embedding` thus transforming the vocab to vectors and then it goes through one `Linear` layer which would be the hidden layer in the neural network.

We also define the `CrossEntropyLoss` which would be the loss function and we use this instead of `MSE` because we are doing classification here, then the `optim.SGD` is the Stochastic Gradient Descent. 

Then in the loop, the training starts, it will run 50 steps so we are doing 50 epochs (runs). In the loop as you can see we transform the values into `tensor`. Then we go through the normal neural network flow:

```bash
The loop does: forward pass → CrossEntropyLoss (which internally applies log-softmax) → backward propagation → SGD step.
```

This keeps looping until the loss result drops then we would have a correct prediction in the model.

Also I'm using `matplotlib` to plot the vectors in the diagram, which you can see when you run it.

## How to Run

Install `uv` first:

```bash
pip install uv
```

Then execute the command `uv sync` which would create the `.venv` and install the packages needed to run the file.
Then to run it execute the command: `uv run neural_network.py`

## Let's look at the math

Now let's look what's happening at each step:

Assume the embedded weights are the following:

```py
tensor([[-0.9937, -0.4648],
        [-0.9480,  1.2738],
        [ 0.4065,  0.6565]])
```

The Linear layer weights are the following:

```py
tensor([[-0.3479,  0.3685],
        [ 0.4141, -0.1088],
        [ 0.1766,  0.1483]])
```

The Linear layer biases are the following:

```py
tensor([-0.5017, -0.5832, -0.2037])
```

So since `dog` is at index 0, then the vector for it is: `[−0.9937, −0.4648]`. Then in the forward pass we have the following formula:

```bash
z=vW⊤+b
```

where `v` is the vector and `W` is the linear layer weights and `b` is the linear layer bias. So to compute the logit for `dog`:

```python
## logit for dog
Row 0 of W = [−0.3479,0.3685] 
z0​ = (−0.9937)(−0.3479)+(−0.4648)(0.3685)−0.5017 # dot product
z0 = −0.3271

## logit for cat
Row 1 of W = [0.4141,−0.1088]
z1 = −0.9445

## logit for park
Row 2 of W = [0.1766,0.1483]
z2 = −0.4482

z = [−0.3271, −0.9445, −0.4482] #logit vector
```

Then the softmax is applied on the logit vector, using the following formula:

```py
softmax(z_j) = e^(z_j) / sum( e^(z_k) )

e^−0.3271 = 0.7211
e^−0.9445 = 0.3893
e^−0.4482 = 0.6391

S = 0.7211 + 0.3893 + 0.6391 = 1.7495

P0 = 0.7211/1.7495 = 0.4121
P1 = 0.3893/1.7495 = 0.2225
P2 = 0.6391/1.7495 = 0.3654
```

`CrossEntropyLoss` uses the probability for the target class (park, index 2):


```python
L = −log(p2​)
L = −log(0.3654) = 1.007
```

After the loss function is computed, then we have the backpropagation, here we compute the gradient layer by layer using the system of chain-rule equations. So we would need to update the gradient w.r.t logits, weights, bias, and embedding vector. Finally we have the stochastic gradient descent which would update the weights and biases and it would have the following formula:

```python
θ_new = θ_old − α * (∂L/∂θ)
```
Where:
- θ = any parameter (weight, bias, embedding)
- α = learning rate
- ∂L/∂θ = gradient of the loss with respect to that parameter
- θ_old = old weight and bias

So to give an example, assuming we have the following computed gradient:

```python
g = [0.4121, 0.2225, -0.6346]       (logit gradients)
v = [-0.9937, -0.4648]              (embedding vector)
∂v = [-0.1632, 0.0334]              (embedding gradient)
```
Learning rate α = 0.1

First we update the embedding vector for the input word `dog`:

Old embedding:

```py
E[0] = [-0.9937,  -0.4648]
```

Embedding gradient:

```py
∂E[0] = [-0.1632,  0.0334]
```

SGD update:

```python
E[0][0] = -0.9937 − 0.1*(-0.1632)
        = -0.9937 + 0.01632
        = -0.97738

E[0][1] = -0.4648 − 0.1*(0.0334)
        = -0.4648 − 0.00334
        = -0.46814
```

New embedding:

```py
E[0] = [-0.97738, -0.46814]
```

Linear weights update W[j][i]:

So assuming the gradients are the following:

```py
∂W[0] = [-0.4094, -0.1918]
∂W[1] = [-0.2211, -0.1034]
∂W[2] = [ 0.6305,  0.2950]
```


And your current linear weights:


```py
W =
[[-0.3479,  0.3685],
 [ 0.4141, -0.1088],
 [ 0.1766,  0.1483]]
```


Apply SGD:

Row 0:

```py
W[0][0] = -0.3479 − 0.1*(-0.4094)
        = -0.3479 + 0.04094
        = -0.30696

W[0][1] =  0.3685 − 0.1*(-0.1918)
        =  0.3685 + 0.01918
        =  0.38768
```

Row 1:

```py
W[1][0] = 0.4141 − 0.1*(-0.2211)
        = 0.4141 + 0.02211
        = 0.43621

W[1][1] = -0.1088 − 0.1*(-0.1034)
        = -0.1088 + 0.01034
        = -0.09846
```

Row 2:

```py
W[2][0] = 0.1766 − 0.1*(0.6305)
        = 0.1766 − 0.06305
        = 0.11355

W[2][1] = 0.1483 − 0.1*(0.2950)
        = 0.1483 − 0.02950
        = 0.11880
```

Then you also update the bias using the same formula as above. Eventually you would have new weights, biases, and vectors for example:

New embedding for “dog”:

```py
[-0.97738, -0.46814]
```

New linear weights:

```py
[[-0.30696,  0.38768],
 [ 0.43621, -0.09846],
 [ 0.11355,  0.11880]]
```

New biases:

```py
[-0.54291, -0.60545, -0.14024]
```

and that's it. This would keep iterating until the 50 steps are done.