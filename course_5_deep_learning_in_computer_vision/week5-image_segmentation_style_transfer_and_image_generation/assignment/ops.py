# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/ops.py
#   + License: MIT

import math
import numpy as np
import tensorflow as tf
from utils import *


class BatchNorm(object):
    """
    Class for doing batch normalization
    
    Code modification of http://stackoverflow.com/a/33950177
    """
    
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        """
        Constructor of the batch normalization class
        
        Parameters
        ----------
        epsilon : float
            A float added to variance to avoid dividing by zero
        momentum :
            Momentum for the moving mean and the moving variance
        name : str
            Name of the variable scope
        """
        
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, is_training=True):
        """
        The calling function of the class
        
        Parameters
        ----------
        x : Tensor, shape (...)
            The input tensor
        is_training : boll
            Whether or the layer is under training
            
        Returns
        -------
        Tensor, shape (...)
            The result of batch normalization
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)

def conv2d(input_,
           output_dim,
           k_h=5,
           k_w=5,
           d_h=2,
           d_w=2,
           stddev=0.02,
           name='conv2d'):
    """
    2D convolutional layer

    Notes
    -----
    The weigths are initialized with a gaussian (truncated) initializer,
    whereas the biases are initialized with a constant value

    Parameters
    ----------
    input_ : Tensor, shape (n_batch, in_rows, in_cols, in_depth)
        The tensor to which the 2d convolution should be applied to
    output_dim : int
        Number of filters
    k_h : int
        Kernel height
    k_w : int
        Kernel width
    d_h : int
        Stride height
    d_w : int
        Stride width
    stddev : float
        The standard deviation to use for the Gaussian initializer for the weights
    name : str
        Name of the variable scope
        
    Return
    ------
    Tensor, shape (n_batch, 1+[in_rows−k_h+2*padding]/d_h, 1+[in_cols−k_w+2*padding]/d_w, output_dim)
        The result of the convolution
    """
    
    with tf.variable_scope(name):
        # NOTE: This is a rank 4 variable
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # NOTE: Stride 1 in batch and 1 in depth
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

    
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

        
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    Performs the matrix operation A*W + b
    
    Notes
    -----
    Do not confuse with linear activation.
    This is equivalent to a forward pass in a dense layer.
    The weigths are initialized with a gaussian initializer, whereas the biases are initialized with a constant value.
    
    Parameters
    ----------
    input_ : array-like, shape (in_rows, in_cols)
        The input array
    output_size : int
        The number of rows for output matrix
    scope : None or str
        Name of the scope of the operations creating the output variable
        (i.e. the name of the layer)
    stddev : float
        The standard deviation to use for the Gaussian initializer for the weights
    bias_start : float
        The value to use for the initialization of the bias
    with_w : bool
        Whether the weights (call matrix) and the bias should be returned
    
    Returns
    -------
    Tensor, shape (in_rows, output_size)
        The result of the A*W + b operation
    matrix : Variable, shape (in_cols, output_size)
        Only returned if with_w is True
        The W in the matrix multiplication
    bias : Variable, shape (output_size)
        Only returned if with_w is True
        The b in the matrix multiplication
    """
    
    shape = input_.get_shape().as_list()

    # NOTE: The variable scope is for sharing variables
    # https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
