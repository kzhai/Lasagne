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

from .layers import get_output, get_output_shape, DenseLayer, LinearDropoutLayer, AdaptiveDropoutLayer

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

def l1_norm(X, axis=None):
    return T.sum(abs(X), axis=axis)

def l2_norm(X, axis=None):
    return T.sqrt(T.sum(X ** 2, axis=axis))

def linf_norm(X, axis=None):
    return T.max(abs(X), axis=axis);

def __find_layer_before_dropout(network, axis=None):
    for layer_1, layer_2 in zip(network.get_network_layers()[1:], network.get_network_layers()[:-1]):
        if isinstance(layer_2, LinearDropoutLayer) or isinstance(layer_2, AdaptiveDropoutLayer):
            return layer_1

def rademacher_p_2_q_2(network, **kwargs):
    #input_shape = get_output_shape(network._input_layer);
    #input_value = get_output(network._input_layer);
    #n = input_shape[0];
    #d = T.prod(input_shape[1:]);
    pseudo_input_layer = __find_layer_before_dropout(network);

    n = network._input_variable.shape[0];
    #d = T.prod(network._input_variable.shape[1:]);
    d = T.prod(get_output_shape(pseudo_input_layer)[1:]);
    dummy, k = network.get_output_shape();
    rademacher_regularization = k * T.sqrt(T.log(d) / n);
    #rademacher_regularization *= T.max(abs(network._input_variable));
    rademacher_regularization *= T.max(abs(get_output(pseudo_input_layer)));

    for layer in network.get_network_layers():
        if isinstance(layer, LinearDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
            retain_probability = T.clip(layer.activation_probability, 0, 1);
            rademacher_regularization *= T.sqrt(T.mean(retain_probability**2))
        elif isinstance(layer, DenseLayer):
            # compute B_l * p_l, with a layer-wise scale constant
            d1, d2 = layer.W.shape
            rademacher_regularization *= T.max(T.sqrt(T.sum(layer.W ** 2, axis=0)))
            rademacher_regularization /= T.sqrt(d1 * T.log(d2))
            # this is to offset glorot initialization
            rademacher_regularization *= T.sqrt((d1 + d2))

    return rademacher_regularization

rademacher = rademacher_p_2_q_2  # shortcut

def rademacher_p_inf_q_1(network, **kwargs):
    n = network._input_variable.shape[0];

    n, d = network._input_variable.shape;
    dummy, k = network.get_output_shape();
    rademacher_regularization = k * T.sqrt(T.log(2 * d) / n);
    rademacher_regularization *= T.max(abs(network._input_variable));

    for layer in network.get_network_layers():
        if isinstance(layer, LinearDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
            #retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1);
            retain_probability = T.clip(layer.activation_probability, 0, 1);
            rademacher_regularization *= T.mean(abs(retain_probability))
        elif isinstance(layer, DenseLayer):
            # compute B_l * p_l, with a layer-wise scale constant
            d1, d2 = layer.W.shape
            rademacher_regularization *= T.max(abs(layer.W));
            # this is to offset glorot initialization
            rademacher_regularization *= T.sqrt((d1 + d2))

    return rademacher_regularization

def rademacher_p_1_q_inf(network, **kwargs):
    n = network._input_variable.shape[0];

    n, d = network._input_variable.shape;
    dummy, k = network.get_output_shape();
    rademacher_regularization = k * T.sqrt(T.log(2 * d) / n);
    rademacher_regularization *= T.max(abs(network._input_variable));

    for layer in network.get_network_layers():
        if isinstance(layer, LinearDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
            #retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1);
            retain_probability = T.clip(layer.activation_probability, 0, 1);
            rademacher_regularization *= T.max(abs(retain_probability));
        elif isinstance(layer, DenseLayer):
            # compute B_l * p_l, with a layer-wise scale constant
            d1, d2 = layer.W.shape
            rademacher_regularization *= T.max(T.sum(abs(layer.W), axis=0))
            rademacher_regularization /= d1 * T.sqrt(T.log(d2));
            # this is to offset glorot initialization
            rademacher_regularization *= T.sqrt((d1 + d2))

    return rademacher_regularization
