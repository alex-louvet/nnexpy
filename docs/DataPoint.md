# DataPoint

`DataPoint` stores a point coordinates as well as its class (`cluster`)

## Class Variables

`dimension` (`int`): Dimension of the `DataPoint`. Corresponds to the length of `coordinates`

`coordinates` (`(float)`): Tuple representing the coordinates of the `DataPoint`. Length corresponds to `dimension`

`cluster` (`int`): Represents the class of the `DataPoint`

## Constructors

```python
__init__(dimension=2, coordinates=(random.random(), random.random()), cluster=0)
```
A data point

## Methods

```python
distanceTo(DataPoint):
  return float
```
Computes the euclidian distance from the `DataPoint` to another one passed in argument
