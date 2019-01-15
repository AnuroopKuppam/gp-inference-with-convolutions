---
 typora-copy-images-to: ./
---

# $\mathcal{GP}$ Write Up 10

### November 6, 2018

#### Checking for machine precision with naive GP

## 1. Experiments

We are trying to determine if the predictive mean from naive GP and the convolutional method are equal.

That is:

Naive:		
$$
\mathbf{f}_* = \mathbf{K}(\mathbf{K}+\sigma_n^2I)^{-1}y
$$
Convolutional method:
$$
v = \mathbf{K}^{-1}\mathbf{f}_*
$$
We used three different smooth surfaces each defined between $X\in[-25,25]$, $Y\in[-25,25]$. We ran the network for 600 iterations, with a learning rate of $10^{-1}$ and we find that the predictive mean for the convolutional method and the naive GP are within machine precision of $10^{-15}$. The observations have noise $\mathcal{N}(0,\sigma_n^2I)$.

| Equation                        | Predictive mean                                              | SE      |
| ------------------------------- | ------------------------------------------------------------ | ------- |
| $\frac{X^2}{4}+\frac{Y^2}{8}$   | ![pred_parabola_0.1](/home/satya/Projects/gps/write ups/pred_parabola_0.1.png) | 1.7e-13 |
| $X+Y$                           | ![pred_plane_0.01](/home/satya/Projects/gps/write ups/pred_plane_0.01.png) | 8.4e-18 |
| $\frac{X^3}{16}+\frac{Y^3}{64}$ | ![pred_double_0.01](/home/satya/Projects/gps/write ups/pred_double_0.01.png) | 9.5e-24 |





