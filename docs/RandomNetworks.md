# RandomNetworks

## Class Variables

`layers` (`[Layer]`): Layers of the network

## Constructors

```python
__init__(inputDimension=2,depth=1,outputDimension=1,width=8):
```
Generates a simple set of weight and bias emulating a Keras + Tensforflow `Sequential` 

## Methods

```python
get_weights():
  return [[float],[float]]:
```
Get the weights and biases of a `Layer`
