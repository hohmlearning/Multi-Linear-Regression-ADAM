# Multi-Linear-Regression-ADAM
In this repository, Linear-Regression is solved by Adaptive Moment Estimation ADAM with early-stopping [1]. 
ADAM is an optimization methode for non-convex loss function. 
Although, Linear-Regression is convex and is solved with Ordinary Least Squares, if the dataset is not too large, ADAM is capable of finding solutions. 
In Particular, early stopping is integrated in ADAM as an regulator. 
Early stopping is the most common used regulator in deep learning and acts similar to $L^2$ regularization [2].
In Addition, early stopping determines the amount of regularization automatically while monitoring the validation loss or metric [2].


Stochastic Gradient Descent (SGD).
Adressing drawbacks of SGD:
- noise gradient to do calculating on batch size\
"The method combines the advantages of
two recently popular optimization methods: the ability of AdaGrad to deal with sparse gradients,
and the ability of RMSProp to deal with non-stationary objectives."
ADAM:\
Adaptive learning, estimate first and second order gradients for high-dimensional parameter spaces
- Momentum: Smooths the gradient
- Momentum$^{2}$: Smooths the momentum; balancing the directions for noisy and / or sparse gradients


## Stochastic gradient descent
## Calculation the gradient
With maximizing the Maximum Likelihood Estimation (MLE), the loss function, Sum of Squared Errors (SSE), is obtained:
$$L(w, w_0; X, y) = ||y - (Xw + w_0)||^2 $$

The expression is rarranged:
$$L(w, w_0; X, y) = (y - (Xw + w_0))^T   (y - (Xw + w_0))$$

$$L(w, w_0; X, y) = y^Ty - y^T(Xw + w_0) - (Xw + w_0)^T y + (Xw + w_0)^T  (Xw + w_0)$$

The loss function is derived with respect to the weight vector $w$ and the bias term $w_0$:
$$\frac{\partial L(w, w_0; X, y)}{\partial w} = -2 X^T y + 2 X^T (X w + w_0)$$
$$ = -2 X^T (y - (X w + w_0))$$
<br>
$$\frac{\partial L(w, w_0; X, y)}{\partial w_0} = - 2 (y^T \mathbb{1} -(Xw + w_0)^T \mathbb{1} )$$
with: $$\mathbb{1} \in R^{n\times1}$$
<br>

## References
[1] D.P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, 2014
[2] I. Goodfellow, Y. Bengio, A. Courville, Deep learning, The MIT Press, Cambridge, Massachusetts, London, England, 2016
