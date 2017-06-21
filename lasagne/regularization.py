"""
Functions to apply regularization to the weights in a network.

We provide functions to calculate the L1 and L2 penalty. Penalty functions
take a tensor as input and calculate the penalty contribution from that tensor:

.. autosummary::
    :nosignatures:

    l1
    l2

A helper function can be used to apply a penalty function to a tensor or a
list of tensors:

.. autosummary::
    :nosignatures:

    apply_penalty

Finally we provide two helper functions for applying a penalty function to the
parameters in a layer or the parameters in a group of layers:

.. autosummary::
    :nosignatures:

    regularize_layer_params_weighted
    regularize_network_params

Examples
--------
>>> import lasagne
>>> import theano.tensor as T
>>> import theano
>>> from lasagne.nonlinearities import softmax
>>> from lasagne.layers import InputLayer, DenseLayer, get_output
>>> from lasagne.regularization import regularize_layer_params_weighted, l2, l1
>>> from lasagne.regularization import regularize_layer_params
>>> layer_in = InputLayer((100, 20))
>>> layer1 = DenseLayer(layer_in, num_units=3)
>>> layer2 = DenseLayer(layer1, num_units=5, nonlinearity=softmax)
>>> x = T.matrix('x')  # shp: num_batch x num_features
>>> y = T.ivector('y') # shp: num_batch
>>> l_out = get_output(layer2, x)
>>> loss = T.mean(T.nnet.categorical_crossentropy(l_out, y))
>>> layers = {layer1: 0.1, layer2: 0.5}
>>> l2_penalty = regularize_layer_params_weighted(layers, l2)
>>> l1_penalty = regularize_layer_params(layer2, l1) * 1e-4
>>> loss = loss + l2_penalty + l1_penalty
"""
import theano.tensor as T
from .layers import Layer, get_all_params


def l1(x):
    """Computes the L1 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        l1 norm (sum of absolute values of elements)
    """
    return T.sum(abs(x))


def l2(x):
    """Computes the squared L2 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        squared l2 norm (sum of squared values of elements)
    """
    return T.sum(x**2)


def linf(x):
    """Computes the Linfinity norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        linf norm (max of absolute values of elements)
    """
    return T.max(abs(x))


def apply_penalty(tensor_or_tensors, penalty, **kwargs):
    """
    Computes the total cost for applying a specified penalty
    to a tensor or group of tensors.

    Parameters
    ----------
    tensor_or_tensors : Theano tensor or list of tensors
    penalty : callable
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the total penalty cost
    """
    try:
        return sum(penalty(x, **kwargs) for x in tensor_or_tensors)
    except (TypeError, ValueError):
        return penalty(tensor_or_tensors, **kwargs)


def regularize_layer_params(layer, penalty,
                            tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of a layer or group of layers.

    Parameters
    ----------
    layer : a :class:`Layer` instances or list of layers.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    layers = [layer, ] if isinstance(layer, Layer) else layer
    all_params = []

    for layer in layers:
        all_params += layer.get_params(**tags)

    return apply_penalty(all_params, penalty, **kwargs)


def regularize_layer_params_weighted(layers, penalty,
                                     tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of a layer or group of layers, weighted by a coefficient for each layer.

    Parameters
    ----------
    layers : dict
        A mapping from :class:`Layer` instances to coefficients.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    return sum(coeff * apply_penalty(layer.get_params(**tags),
                                     penalty,
                                     **kwargs)
               for layer, coeff in layers.items()
               )


def regularize_network_params(layer, penalty,
                              tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of all layers in a network.

    Parameters
    ----------
    layer : a :class:`Layer` instance.
        Parameters of this layer and all layers below it will be penalized.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    return apply_penalty(get_all_params(layer, **tags), penalty, **kwargs)

#
#
#
#
#

import numpy
from .layers import get_output, DenseLayer, LinearDropoutLayer, AdaptiveDropoutLayer

def regularize_layer_weighted(layers, penalty, tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to a group of layers, weighted by a coefficient for each layer.

    Parameters
    ----------
    layers : dict
        A mapping from : tuple of class:`Layer` instances to coefficients.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
        Should be defined inside the penalty function
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    return sum(coeff * sum(penalty(layer_tuple, tags, **kwargs)) for layer_tuple, coeff in layers.items());

def rademacher(network, **kwargs):
    rademacher_regularization = linf(network._input_variable);
    #print network._input_layer;
    #print network._input_variable;

    first_hidden_layer = True;
    #dense_layers = [];
    for layer in network.get_network_layers():
        if first_hidden_layer:
            if isinstance(layer, DenseLayer):
                #print layer.W.eval().shape
                #print T.sum(abs(layer.W), axis=0).eval().shape
                #print "layer parameter:", T.sum(abs(layer.W), axis=0)
                rademacher_regularization *= T.max(T.sum(abs(layer.W), axis=0))
                first_hidden_layer = False;
        else:
            if isinstance(layer, LinearDropoutLayer):
                retain_probability = layer.activation_probability;
            elif isinstance(layer, DenseLayer):
                #print "retain probability shape:", retain_probability.shape
                d1, d2 = layer.W.eval().shape
                rademacher_regularization *= linf(layer.W) / numpy.sqrt(numpy.log(d2)) * sum(retain_probability) / d1;
    return rademacher_regularization

def l1_norm(layer, tags={'regularizable': True}, **kwargs):
    """Computes the L1 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        l1 norm (sum of absolute values of elements)
    """
    return T.sum(abs(layer.get_params(**tags)))

def l2_norm(layer, tags={'regularizable': True}, **kwargs):
    """Computes the squared L2 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        squared l2 norm (sum of squared values of elements)
    """
    return T.sum(layer.get_params(**tags)**2)

def linf_norm(layer, tags={'regularizable': True}, **kwargs):
    """Computes the L1 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        l1 norm (sum of absolute values of elements)
    """
    return T.max(abs(layer.get_params(**tags)))