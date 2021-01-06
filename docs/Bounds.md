# Bounds

## Class Variables

`boundCoordinates` (`[{'min': float , 'max': float}]`): Represents the minimum and maximum of the bounds. Length corresponds to `dimension`

`dimension` (`int`): Dimension of the bounds. Corresponds to the length of `boundCoordinates`

## Constructors

```python
__init__(dimension=2, boundCoordinates=[{'min': 0, 'max' : '1'}, {'min': 0, 'max' : '1'}])
```
If both `dimension` and `boundCoordinates` are specified they must correspond according to the explanation in Class Variables.

## Methods

