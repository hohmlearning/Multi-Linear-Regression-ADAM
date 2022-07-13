# Linear-Regression-ADAM
In this repository, Linear-Regression is solved by Adaptive Moment Estimation ADAM with early-stopping [1]. 
ADAM is an optimization methode for non-convex loss function. 
Although, Linear-Regression is convex and is solved with Ordinary Least Squares, if the dataset is not too large, ADAM is capable of finding solutions. 
In Particular, early stopping is integrated in ADAM as an regulator. 
Early stopping is the most common used regulator in deep learning and acts similar to $L^2$ regularization [2].
In Addition, early stopping determines the amount of regularization automatically while monitoring the validation loss or metric [2].\
In future, Linear-Regression ought to be compared with Ridge-Regression (ODE with $L^2$ regularization).

## ADAM
ADAM is a first order optimizer estimating first and second order gradients [1]. 
ADAM is a combination of AdaGrad, which is capable of sparse gradients, and RMSProp, which suits non-stationary objectives [1]. 
In general, stochastic optimizers refer to partial gradients.
For fitting the gradient in the memory, batches are drawn from dataset randomly.
Due to calculation the partial gradients, noise is introduced in the gradient.
For smoothing the gradient, the gradient is estimated based on the momentum.
For balancing the directions for sparse or noisy gradients, the momentum is smoothed with the average past squared gradients.

## Algorithm
The ADAM algorithm is shown in **Figure 1**.

![grafik](https://user-images.githubusercontent.com/107933496/178591917-e638d6eb-db14-44fd-9b6d-8bdc5c681a57.png)

**Figure 1**: ADAM algorithmus copied from Kignma and Ba [1].

The weight vector $w$ and the bias term $w_0$ is initialized to 0. 
Exponential moving average is applied to the gradient and the squared gradient resulting in first $m_t$ and second moment vector $v_t$, respectively.
The parameters for the exponential average are $\beta_1$ and $\beta_2$.
First $m_t$ and second moment vector $v_t$ are initialized to 0.
The bias-correction is applied to overcome slow convergence in the first steps $t$.
For details, the author refers to the original paper [1].

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

## Learning rate decay
Although ADAM is an adaptive learning algorithm, the minimal loss function is archive with the optimal learning rate.
Thereby, the optimal learning rate is model and problem specific. 
In analogy to Zhang and Sun [3], the learning rate is changed, whenever the error and / or loss plateaus.\
When **ADAM_learning_rate_decay** is deployed, the optimal paramters (weight $w$ and bias $w_0$) and the learning rate decay are validated consulting the error. 
If a validation set is given when calling, the error of the validation data is kept track.
Otherwise, the trainings error is consulted. 
At the end of each epoch, the parameters are saved and the **patience counter** is set to 0, if an optimal error was archived.
Whenever no improvement of the error after an epoch is made, the **patience counter** is increased by 1.
The learning rate is decayed, when the **patience counter** equals the patience set with the call.
The new learning rate is adapted:

$$
\eta <- \frac{\eta}{\sqrt{10}}
$$



## Early Stopping
The trainings error is decreasing with the number of data the model has seen.
Unfortunately, overfitting, the increase in validation and test error while decreasing the trainings error, is liekly to occur.
Thus, the model capability to generalize is decreasing.
Early stopping is a methode to overcome overfitting acting similar to the $L^2$ regularization [2].\
Internally, the trainings data set is splitted into subtrain (80 %) and validation (20 %) data set.
While applying early stopping, after each 



## References
[1] D.P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, 2014\
[2] I. Goodfellow, Y. Bengio, A. Courville, Deep learning, The MIT Press, Cambridge, Massachusetts, London, England, 2016\
[3] K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, 2015
