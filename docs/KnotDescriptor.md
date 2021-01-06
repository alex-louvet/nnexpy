# KnotDescriptor

## Class Variables

`crossing` (`[int]`): index of points at the crossing of the knot (last point of the component before the crossing)

`component` (`[(int, int)]`): index of the points at the beginning and end of the components of the knot

`trajectory` ([(float)]): list of the points composing the knot

## Constructors

```python
__init__(crossing, component, trajectory=[])
```
Generates a knot with no trajectory

---

```python
fromTrajectory(trajectory)
```
Generates a knot from a trajectory

---

```python
fromTemplate(template, nPoints=1000, radius=1, center=0)
```

Generates a knot according to a template, available templates are:
* `'circular3D'`
* `'circular4D'`
* `'trefoil3D'`
* `'trefoil4D'`
* `'figureEight3D'`
* `'figureEight4D'`

## Methods

```python
findCrossingComponent(crossing):
  return (int, int, int)
```
Give the index in `component` of the components involved in the crossing (must be an element of `crossing`). The first index of the result is the top component

---

```python
determinant():
  return float
```
Computes the determinant of the `KnotDescriptor`

---

```python 
plotTraj2D():
```
Projects the trajectory in 2 dimension and plot it highlighting the left-hand knot in orange and the right-hand knot in blue (`matplotlib` default colors)

---

```python
plotTraj3D():
```
Projects (if necessary) the trajectory in 3 dimension and plot it highlighting the left-hand knot in orange and the right-hand knot in blue (`matplotlib` default colors)
