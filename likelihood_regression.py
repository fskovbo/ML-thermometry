import tensorflow as tf
import numpy as np

def weibull_loss(y_true, y_pred):
    """
    Loss function for Weibull distribution.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        k and l values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    eps = 1e-07

    # Separate the parameters
    k, la = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    k = tf.expand_dims(k, -1)
    la = tf.expand_dims(la, -1)
    
    # Calculate the negative log likelihood
    nll = (
        tf.pow(tf.math.divide(y_true, la+eps), k)
        + k * tf.math.log(la+eps)
        - k * tf.math.log(y_true)
        - tf.math.log(k+eps)
        + tf.math.log(y_true)
    )                  

    return tf.reduce_mean(nll) # take mean over all samples in batch


def normal_loss(y_true, y_pred):
    """
    Loss function for normal distribution.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    eps = 1e-5

    # Separate the parameters
    mu, sigma = tf.unstack(y_pred, num=2, axis=-1)
    
    # Add one dimension to make the right shape
    mu = tf.expand_dims(mu, -1)
    sigma = tf.expand_dims(sigma, -1)
    
    # Calculate the negative log likelihood
    nll = (
        0.5*tf.math.log(2*np.pi)
        + 0.5*tf.math.log( tf.math.pow(sigma+eps,2) )
        + 0.5*tf.math.multiply( tf.math.pow(sigma+eps,-2), tf.math.pow(y_true-mu,2) )
    )               

    return tf.reduce_mean(nll) # take mean over all samples in batch


def normal_rmse(y_true, y_pred):
    """
    Metric function for normal distribution.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma values of predicted distribution.
        
    Returns
    -------
    metric : tf.Tensor
        mse between mu and y_true.
    """

    # Separate the parameters
    mu, sigma = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    mu = tf.expand_dims(mu, -1)
    sigma = tf.expand_dims(sigma, -1)

    metric = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(mu-y_true)))

    return metric