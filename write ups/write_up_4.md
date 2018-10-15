---
 typora-copy-images-to: ./
---

# $\mathcal{GP}$ Write Up 4

### September 18, 2018

#### Experimental results: Convolution in Tensorflow with missing observations

Code: https://colab.research.google.com/drive/1CJGTX0UqPs0QVPQ-6FbnhJmAA_BukxZ1

## 1. Missing Observations on a Smooth Surface

Three dimensional Smooth Surface	 $Z = \frac{X^2}{4} + \frac{Y^2}{8}$ with $X \in [-0.25, 0.25]$ and $Y \in [-0.25, 0.25]$. 

Kernel trained using a 2D convolution layer, with a squared exponential kernel with $\theta=[\sigma_f^2, l]$  and $\sigma_f=0.1$ and $l=2$.  

| % missing Observations | iterations | variance | loss            | learning rate |                            Image                             | Error |
| ---------------------- | ---------- | -------- | --------------- | ------------- | :----------------------------------------------------------: | ----- |
| 2                      | 3k         | 0        | $9*10^{-4}$     | 0.01          | ![variance_0](/home/satya/Projects/gps/write ups/variance_0.png) |       |
| 2                      | 3k         | 1.0      | $1.8*10^{-5}$   | 0.01          | ![variance_1_2](/home/satya/Projects/gps/write ups/variance_1_2.png) |       |
| 2                      | 3k         | 10.0     | $5.6*10^{-5}$   | 0.01          | ![variance_10_2](/home/satya/Projects/gps/write ups/variance_10_2.png) |       |
| 30                     | 10k        | 1.0      | $4.2 * 10^{-5}$ | 0.01          |  ![variance_1_30](/home/satya/Downloads/variance_1_30.png)   |       |
| 50                     | 15k        | 1.0      | $3.5 * 10^{-5}$ | 0.01          | ![variance_1_50](/home/satya/Projects/gps/write ups/variance_1_50.png) |       |
| 70                     | 80k        | 1.0      | 230             | 0.1           | ![variance_1_70_80k](/home/satya/Projects/gps/write ups/variance_1_70_80k.png) |       |
| 70                     | 80k        | 10.0     | 0.10            | 0.1           | ![variance_10_70_80k](/home/satya/Projects/gps/write ups/variance_10_70_80k.png) |       |

Three dimensional Smooth Surface	 $Z = X^3 + Y^3$ with $X \in [-0.25, 0.25]$ and $Y \in [-0.25, 0.25]$. 

| % missing Observations | iterations | variance | loss          | learning rate | Image                                                 |
| ---------------------- | ---------- | -------- | ------------- | ------------- | ----------------------------------------------------- |
| 15                     | 10k        | 0.0      | 0.08          | 0.01          | ![double_0_10](/home/satya/Downloads/double_0_10.png) |
| 15                     | 10k        | 1.0      | $2*10^{-5}$   | 0.01          | ![double_1_10](/home/satya/Downloads/double_1_10.png) |
| 30                     | 10k        | 1.0      | $5*10^{-5}$   | 0.01          | ![double_1_30](/home/satya/Downloads/double_1_30.png) |
| 52                     | 10k        | 1.0      | $1.4*10^{-4}$ | 0.01          | ![double_1_52](/home/satya/Downloads/double_1_52.png) |
| 75                     | 10k        | 1.0      | $3.2*10^{-4}$ | 0.01          | ![double_1_74](/home/satya/Downloads/double_1_74.png) |

