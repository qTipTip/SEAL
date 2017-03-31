# SEAL: SplinE Algorithm Library

![A Seal!](images/seal_spline.pdf)

### Introduction

SEAL is a library for working with spline functions in a quick intuitive way.
It provides callable SplineFunction object, as well as SplineSpace objects that
function mainly as wrappers around the SplineFunctions. SEAL makes quickly getting up and running
with splines super easy!

### Features

SEAL currently supports:

1. Scalar Spline Functions;
2. Parametric Spline Functions in arbitrary dimensions (Euclidean space of dimension d);
3. Retrieving the control polygon of a SplineFunction for ease of plotting.
4. Compute the Variation Diminishing Spline Approximation of a scalar / parametric function.
5. Finding the Cubic Hermite Spline Interpolant to a set of data points, with derivatives supplied.
6. Computing the Least Square Spline Approximation to a set of data points in a given spline space.
