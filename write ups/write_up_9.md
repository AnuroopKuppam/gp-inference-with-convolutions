---
 typora-copy-images-to: ./
---

# $\mathcal{GP}$ Write Up 9

### October 31, 2018

#### Nature of the latent variable $v$ 

## 1. How $v$ differs wrt to the kernel function

For the experiments shown below consider the signal below which is a mixture of three different sine waves:

![signal](/home/satya/Projects/gps/write ups/signal.png)

				              **Figure 1**: The signal($y$) we are trying to approximate
	
	             
	             ![signal_freq_domain](/home/satya/Projects/gps/write ups/signal_freq_domain.png)  
	
						**Figure 2:** The signal($y$) in the frequency domain



The signal($y$) is noise free and we are interested in $v$ such that $y =Kv$ where $K$ is the covariance matrix of the signal. 

For the covariance function we choose the squared exponential $k(x,y) = \sigma_f^2 exp(\frac{-||x-y||^2_2}{2l^2})$.

| $\sigma_f, l$ | $K$ in freq domain                                           | $v$                                                        | $v$ in freq domain                                           |
| ------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| 5, 0.01       | ![sigma_f_0.01](/home/satya/Projects/gps/write ups/sigma_f_0.01.png) | ![v_0.01](/home/satya/Projects/gps/write ups/v_0.01.png)   | ![v_f_0.01](/home/satya/Projects/gps/write ups/v_f_0.01.png) |
| 5, 0.001      | ![sigma_0.001](/home/satya/Projects/gps/write ups/sigma_0.001.png) | ![v_0.001](/home/satya/Projects/gps/write ups/v_0.001.png) | ![v_f_0.001](/home/satya/Projects/gps/write ups/v_f_0.001.png) |
| 1, $\sqrt{2}$ | ![sigma_f_1](/home/satya/Projects/gps/write ups/sigma_f_1.png) | ![v_1](/home/satya/Projects/gps/write ups/v_1.png)         | ![v_f_1](/home/satya/Projects/gps/write ups/v_f_1.png)       |

From the above table only the first two kernels were able to recreate the signal $y$ with machine precision(tolerance of $10^{-15}$). These kernels have a wide spectrum and hence were able to approximate all the frequency components from our input signal. Infact for second case the spectrum is wide enough that $v$ is similar to the input signal. What this means is that the kernel $k$ should atleast encopass the frequency spectrum of the input signal. This is the reason why we were not able to achieve machine precision before.