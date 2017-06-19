import theano
import theano.tensor as T

from .base import Layer
from ..random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "DropoutLayer",
    "dropout",
    "dropout_channels",
    "spatial_dropout",
    "dropout_locations",
    "GaussianNoiseLayer",
    #
    #
    #
    #
    #
    "validate_activation_probability",
    "sample_activation_probability",
    "AdaptiveDropoutLayer",
    "TrainableDropoutLayer",
    "LinearDropoutLayer",
    "GaussianDropoutLayer",
    "FastDropoutLayer",
    "VariationalDropoutLayer",
    "VariationalDropoutTypeALayer",
    "VariationalDropoutTypeBLayer",
    #
    "BetaBernoulliDropoutLayer",
    #
    #"check_p_backup",
    "GeneralizedDropoutLayerBackup",
]


class DropoutLayer(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.

    The behaviour of the layer depends on the ``deterministic`` keyword
    argument passed to :func:`lasagne.layers.get_output`. If ``True``, the
    layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.

    See also
    --------
    dropout_channels : Drops full channels of feature maps
    spatial_dropout : Alias for :func:`dropout_channels`
    dropout_locations : Drops full pixels or voxels of feature maps

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask

dropout = DropoutLayer  # shortcut


def dropout_channels(incoming, *args, **kwargs):
    """
    Convenience function to drop full channels of feature maps.

    Adds a :class:`DropoutLayer` that sets feature map channels to zero, across
    all locations, with probability p. For convolutional neural networks, this
    may give better results than independent dropout [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    *args, **kwargs
        Any additional arguments and keyword arguments are passed on to the
        :class:`DropoutLayer` constructor, except for `shared_axes`.

    Returns
    -------
    layer : :class:`DropoutLayer` instance
        The dropout layer with `shared_axes` set to drop channels.

    References
    ----------
    .. [1] J. Tompson, R. Goroshin, A. Jain, Y. LeCun, C. Bregler (2014):
           Efficient Object Localization Using Convolutional Networks.
           https://arxiv.org/abs/1411.4280
    """
    ndim = len(getattr(incoming, 'output_shape', incoming))
    kwargs['shared_axes'] = tuple(range(2, ndim))
    return DropoutLayer(incoming, *args, **kwargs)

spatial_dropout = dropout_channels  # alias


def dropout_locations(incoming, *args, **kwargs):
    """
    Convenience function to drop full locations of feature maps.

    Adds a :class:`DropoutLayer` that sets feature map locations (i.e., pixels
    or voxels) to zero, across all channels, with probability p.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    *args, **kwargs
        Any additional arguments and keyword arguments are passed on to the
        :class:`DropoutLayer` constructor, except for `shared_axes`.

    Returns
    -------
    layer : :class:`DropoutLayer` instance
        The dropout layer with `shared_axes` set to drop locations.
    """
    kwargs['shared_axes'] = (1,)
    return DropoutLayer(incoming, *args, **kwargs)


class GaussianNoiseLayer(Layer):
    """Gaussian noise layer.

    Add zero-mean Gaussian noise of given standard deviation to the input [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Standard deviation of added Gaussian noise

    Notes
    -----
    The Gaussian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.

    References
    ----------
    .. [1] K.-C. Jim, C. Giles, and B. Horne (1996):
           An analysis of noise in recurrent neural networks: convergence and
           generalization.
           IEEE Transactions on Neural Networks, 7(6):1424-1438.
    """
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            return input + self._srng.normal(input.shape,
                                             avg=0.0,
                                             std=self.sigma)

#
#
#
#
#

import sys
import numpy

from .. import init, nonlinearities

import warnings

from . import Layer, MergeLayer, DenseLayer
from . import get_all_layers

def validate_activation_probability(activation_probability):
    """
    Thanks to our logit parameterisation we can't accept p of greater than or
    equal to 0.5 (or we get inf logitalphas). So we'll just warn the user and
    scale it down slightly.
    """
    assert type(activation_probability)==float;
    if activation_probability >= 1:
        warnings.warn("activation probability is limited between (0, 1), setting to 1-1e-9.", RuntimeWarning)
        return 1. - 1e-6;
    elif activation_probability <= 0:
        warnings.warn("activation probability is limited between (0, 1), setting to 1e-9.", RuntimeWarning)
        return 1e-6;
    else:
        return activation_probability

def sample_activation_probability(input_dimensions, activation_style, activation_parameter):
    activation_probability = None;
    if activation_style == "uniform":
        activation_probability = numpy.random.random(size=input_dimensions);
    elif activation_style == "bernoulli":
        activation_probability = numpy.zeros(input_dimensions) + activation_parameter;
    elif activation_style == "beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        activation_probability = numpy.random.beta(shape_alpha, shape_beta, size=input_dimensions);
    elif activation_style == "reciprocal_beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        input_number_of_neurons = numpy.prod(input_dimensions);
        activation_probability = numpy.zeros(input_number_of_neurons);
        ranked_shape_alpha = shape_alpha / numpy.arange(1, input_number_of_neurons + 1);
        for index in xrange(input_number_of_neurons):
            activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta);
        activation_probability = numpy.reshape(activation_probability, input_dimensions);
    elif activation_style == "reverse_reciprocal_beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        ranked_shape_alpha = shape_alpha / numpy.arange(1, input_dimensions + 1)[::-1];
        input_number_of_neurons = numpy.prod(input_dimensions);
        activation_probability = numpy.zeros(input_number_of_neurons);
        for index in xrange(input_number_of_neurons):
            activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta);
        activation_probability = numpy.reshape(activation_probability, input_dimensions);
    elif activation_style == "mixed_beta_bernoulli":
        beta_mean, shape_beta = activation_parameter;
        scale = beta_mean / (1. - beta_mean);
        input_number_of_neurons = numpy.prod(input_dimensions);
        activation_probability = numpy.zeros(input_number_of_neurons);
        for index in xrange(input_number_of_neurons):
            rank = index + 1;
            activation_probability[index] = numpy.random.beta(rank * scale / shape_beta, rank / shape_beta);
        activation_probability = numpy.reshape(activation_probability, input_dimensions);
    elif activation_style == "geometric":
        input_number_of_neurons = numpy.prod(input_dimensions);
        activation_probability = numpy.zeros(input_number_of_neurons);
        for index in xrange(input_number_of_neurons):
            rank = index + 1;
            activation_probability[index] = (activation_parameter - 1) / numpy.log(activation_parameter) * (
            activation_parameter ** rank)
        activation_probability = numpy.clip(activation_probability, 0., 1.);
        activation_probability = numpy.reshape(activation_probability, input_dimensions);
    elif activation_style == "reciprocal":
        activation_probability = activation_parameter / numpy.arange(1, input_dimensions + 1);
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    elif activation_style == "exponential":
        activation_probability = activation_parameter / numpy.arange(1, input_dimensions + 1);
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    else:
        sys.stderr.write("error: unrecognized configuration...\n");
        sys.exit();

    return activation_probability.astype(numpy.float32)

def get_filter(input_shape, retain_probability, rng=RandomStreams()):
    filter = rng.binomial(size=input_shape, n=1, p=retain_probability, dtype=theano.config.floatX);

    #mask = rng.binomial(mask_shape, p=retain_prob, dtype=input.dtype)

    '''
    if isinstance(input_shape, tuple):
        filter = numpy.zeros(input_shape);
        for dim in xrange(len(retain_probability)):
            filter[:, dim] = self._srng.binomial(size=len(input_shape[0]), n=1, p=retain_probability[dim], dtype=theano.config.floatX)
    else:
        if isinstance(input_shape, T.Variable) and input_shape.ndim == 1:
            #filter = rng.binomial(size=input_shape, n=1, p=0.5, dtype=theano.config.floatX);
            filter = self._srng.normal(size=input_shape, avg=0.0, std=1.0, dtype=theano.config.floatX);
    '''

    return filter

class AdaptiveDropoutLayer(Layer):
    def __init__(self,
                 incoming,
                 num_units,
                 W=init.GlorotUniform(gain=init.GlorotUniformGain[nonlinearities.sigmoid]),
                 b=init.Constant(0.),
                 alpha=1,
                 beta=0,
                 rescale=True,
                 **kwargs):
        super(AdaptiveDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        self.num_units = num_units
        num_inputs = int(numpy.prod(self.input_shape[1:]))

        self.alpha = alpha
        self.beta = beta

        self.W = self.add_param(W, (num_inputs, num_units), name="W", adaptable=True)
        self.b = self.add_param(b, (num_units,), name="b", regularizable=False, adaptable=True);
        #self.W = self.add_param(W, (num_inputs, num_units), name="W")
        #self.b = self.add_param(b, (num_units,), name="b", regularizable=False);

        self.rescale = rescale

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        """
        if deterministic:
            return get_filter((input.shape[0], self.num_units), 1.0, rng=RandomStreams());

        # layer_signal = T.mul(self.alpha, T.dot(input, self.W)) + self.beta
        layer_signal = T.dot(input, self.W);
        if self.b is not None:
            layer_signal = layer_signal + self.b.dimshuffle('x', 0)
        layer_signal = T.mul(self.alpha, layer_signal) + self.beta;
        activation_probability = nonlinearities.sigmoid(layer_signal);

        activation_flag = get_filter((input.shape[0], self.num_units), activation_probability, rng=RandomStreams());
        if self.rescale:
            activation_flag = activation_flag / activation_probability;

        return activation_flag;

    '''
    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic:
            return input

        activation_probability = self.get_output_for(input, **kwargs);
        if numpy.all(activation_probability == 1):
            return input
        else:
            retain_prob = activation_probability
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * get_filter(input_shape, retain_prob, rng=RandomStreams())

    def get_activation_probability(self, input, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        """
        activation_probability = T.dot(input, self.W)
        if self.b is not None:
            activation_probability = activation_probability + self.b.dimshuffle('x', 0)
        return activation_probability
    '''

class TrainableDropoutLayer(Layer):
    """Trainable Dropout layer
    """
    def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)),
                 num_leading_axes=1, shared_axes=(), **kwargs):
        super(TrainableDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        num_inputs = int(numpy.prod(self.input_shape[num_leading_axes:]))
        #self.activation_probability = theano.shared(value=activation_probability, );
        self.activation_probability = self.add_param(activation_probability, (num_inputs,), name="r", trainableDropout=True);

        #self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

        '''
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)
        '''

    def get_output_for(self, input, deterministic=False, **kwargs):
        #if deterministic or self.activation_probability == 1:
        if deterministic:
            return input * self.activation_probability;
        else:
            retain_prob = self.activation_probability.eval();
            retain_prob = numpy.clip(retain_prob, 0, 1);

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask

class LinearDropoutLayer(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.

    The behaviour of the layer depends on the ``deterministic`` keyword
    argument passed to :func:`lasagne.layers.get_output`. If ``True``, the
    layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.

    See also
    --------
    dropout_channels : Drops full channels of feature maps
    spatial_dropout : Alias for :func:`dropout_channels`
    dropout_locations : Drops full pixels or voxels of feature maps

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, activation_probability=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(LinearDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.activation_probability = activation_probability
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        #if deterministic or self.activation_probability == 1:
        if deterministic or numpy.all(self.activation_probability == 1):
            return input
        else:
            # Using theano constant to prevent upcasting
            retain_prob = self.activation_probability
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask

class GaussianDropoutLayer(Layer):
    """
    Replication of the Gaussian dropout of Srivastava et al. 2014 (section
    10). Applies noise to the activations prior to the weight matrix
    according to equation 11 in the Variational Dropout paper; to match the
    adaptive dropout implementation.

    Uses some of the code and comments from the Lasagne GaussianNoiseLayer:
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or tensor scalar, effective dropout probability
    """

    def __init__(self, incoming, activation_probability=0.5, shared_axes=(), **kwargs):
        super(GaussianDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        activation_probability = validate_activation_probability(activation_probability);
        #self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

        self.alpha = numpy.sqrt((1 - activation_probability) / activation_probability);

        '''
        self.sigma = theano.shared(
            value=numpy.array(_logit(numpy.sqrt(p / (1. - p)))).astype(theano.config.floatX),
            name='logitalpha'
        )
        '''

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or T.mean(self.alpha).eval() == 0:
            return input
        else:
            # use nonsymbolic shape for dropout mask if possible
            perturbation_shape = self.input_shape
            if any(s is None for s in perturbation_shape):
                perturbation_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                perturbation_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(perturbation_shape))
            perturbation = 1 + self.alpha * self._srng.normal(input.shape, avg=0.0, std=1.)

            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in perturbation_shape)
                perturbation = T.patternbroadcast(perturbation, bcast)
            return input * perturbation

    '''
    def old__init__(self, incoming, p=0.5, **kwargs):
        super(GaussianDropoutLayer, self).__init__(incoming, **kwargs)
        p = check_p_backup(p)
        self.logitalpha = theano.shared(
                value=numpy.array(_logit(numpy.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )

    def old_get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        self.alpha = T.nnet.sigmoid(self.alpha)
        if deterministic or T.mean(self.alpha).eval() == 0:
            return input
        else:
            return input + \
                input*self.alpha*self._srng.normal(input.shape, avg=0.0, std=1.)
    '''

class FastDropoutLayer(MergeLayer):
    """
    Replication of the Gaussian dropout of Wang and Manning 2012.
    This layer will only work after a dense layer, but can probably be extended
    to work with convolutional layers. Internally, this pulls out the weights
    from the previous dense layer and applies them again itself, throwing away
    the expression passed from the dense layer. This is necessary because we
    need the expression before the nonlinearity is applied and because we need
    to calculate the sigma. This idiosyncratic method was chosen because it
    keeps the dropout architecture descriptions easy to read.

    Uses some of the code and comments from the Lasagne GaussianNoiseLayer:
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or tensor scalar, effective dropout probability
    nonlinearity : a nonlinearity to apply after the noising process
    """
    def __init__(self, incoming, activation_probability=0.5, **kwargs):
        incoming_input = get_all_layers(incoming)[-2]
        MergeLayer.__init__(self, [incoming, incoming_input], **kwargs)
        activation_probability = validate_activation_probability(activation_probability)
        self.alpha = numpy.sqrt((1 - activation_probability) / activation_probability);
        #self.alpha = (1 - activation_probability) / activation_probability;

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        # and store the parameters of the previous layer
        self.num_units = incoming.num_units
        self.theta = incoming.W
        self.b = incoming.b
        self.nonlinearity = incoming.nonlinearity

    def get_output_shape_for(self, input_shapes):
        """
        Output shape will always be equal the shape coming out of the dense
        layer previous.
        """
        return (input_shapes[1][0], self.num_units)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        # repeat check from DenseLayer
        if inputs[1].ndim > 2:
            # flatten if we have more than 2 dims
            inputs[1].ndim = inputs[1].flatten(2)
        mu_z = T.dot(inputs[1], self.theta) + self.b.dimshuffle('x', 0)
        if deterministic or T.mean(self.alpha).eval() == 0:
            return self.nonlinearity(mu_z)
        else:
            # sample from the Gaussian that dropout would produce
            sigma_z = T.sqrt(T.dot(T.square(inputs[1]),
                                   self.alpha*T.square(self.theta)))
            randn = self._srng.normal(size=inputs[0].shape, avg=0.0, std=1.)
            return self.nonlinearity(mu_z + sigma_z*randn)

    '''
    def old__init__(self, incoming, p=0.5, **kwargs):
        incoming_input = get_all_layers(incoming)[-2]
        MergeLayer.__init__(self, [incoming, incoming_input],
                            **kwargs)
        # store p in logit space
        p = check_p_backup(p)
        self.logitalpha = theano.shared(
            value=numpy.array(_logit(numpy.sqrt(p / (1. - p)))).astype(theano.config.floatX),
            name='logitalpha'
        )

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        # and store the parameters of the previous layer
        self.num_units = incoming.num_units
        self.theta = incoming.W
        self.b = incoming.b
        self.nonlinearity = incoming.nonlinearity

    def old_get_output_shape_for(self, input_shapes):
        """
        Output shape will always be equal the shape coming out of the dense
        layer previous.
        """
        return (input_shapes[1][0], self.num_units)

    def old_get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        # repeat check from DenseLayer
        if inputs[1].ndim > 2:
            # flatten if we have more than 2 dims
            inputs[1].ndim = inputs[1].flatten(2)
        self.alpha = T.nnet.sigmoid(self.logitalpha)
        mu_z = T.dot(inputs[1], self.theta) + self.b.dimshuffle('x', 0)
        if deterministic or T.mean(self.alpha).eval() == 0:
            return self.nonlinearity(mu_z)
        else:
            # sample from the Gaussian that dropout would produce
            sigma_z = T.sqrt(T.dot(T.square(inputs[1]),
                                   self.alpha * T.square(self.theta)))
            randn = self._srng.normal(size=inputs[0].shape, avg=0.0, std=1.)
            return self.nonlinearity(mu_z + sigma_z * randn)
    '''

class VariationalDropoutLayer(Layer):
    """
    Base class for variational dropout layers, because the noise sampling
    and initialisation can be shared between type A and B.
    Inits:
        * p - initialisation of the parameters sampled for the noise
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, activation_probability=0.5, adaptive=None, nonlinearity=None, **kwargs):
        super(VariationalDropoutLayer, self).__init__(incoming, **kwargs)
        self.init_adaptive(adaptive, activation_probability=activation_probability)

    def init_adaptive(self, adaptive, activation_probability):
        """
        Initialises adaptive parameters.
        """
        if not hasattr(self, 'input_shape'):
            self.input_shape = self.input_shapes[0]

        self.adaptive = adaptive
        activation_probability = validate_activation_probability(activation_probability)
        # init based on adaptive options:
        if self.adaptive == None:
            # initialise scalar param, but don't register it
            #self.alpha = (1 - activation_probability) / activation_probability;
            alpha = theano.shared(value=numpy.array((1 - activation_probability) / activation_probability).astype(theano.config.floatX))
            self.alpha = alpha;
        elif self.adaptive == "layerwise":
            # initialise scalar param, allow updates
            # alpha = theano.shared(value=numpy.array((1 - activation_probability) / activation_probability).astype(theano.config.floatX))
            alpha = numpy.array((1 - activation_probability) / activation_probability).astype(theano.config.floatX);
            self.alpha = self.add_param(alpha, (), name="variational.dropout.alpha", regularizable=False, adaptable=True);
        elif self.adaptive == "elementwise":
            # initialise param for each activation passed
            # alpha = numpy.array(numpy.ones(self.input_shape[1:]) * (1 - activation_probability) / activation_probability).astype(theano.config.floatX);
            alpha = theano.shared(value=numpy.array(numpy.ones(self.input_shape[1:]) * (1 - activation_probability) / activation_probability).astype(theano.config.floatX))
            self.alpha = self.add_param(alpha, self.input_shape[1:], name="variational.dropout.alpha", regularizable=False, adaptable=True)
        elif self.adaptive == "weightwise":
            # this will only work in the case of dropout type B
            thetashape = (self.input_shapes[1][1], self.input_shapes[0][1])
            # alpha = numpy.array(numpy.ones(thetashape) * (1 - activation_probability) / activation_probability).astype(theano.config.floatX),
            alpha = theano.shared(value=numpy.array(numpy.ones(thetashape) * (1 - activation_probability) / activation_probability).astype(theano.config.floatX))
            self.alpha = self.add_param(alpha, thetashape, name="variational.dropout.alpha", regularizable=False, adaptable=True);
        else:
            sys.stderr.write("error: unrecognized configuration...\n");
            sys.exit();

    '''
    def old_init_adaptive(self, adaptive, p):
        """
        Initialises adaptive parameters.
        """
        if not hasattr(self, 'input_shape'):
            self.input_shape = self.input_shapes[0]
        self.adaptive = adaptive
        p = check_p_backup(p)
        # init based on adaptive options:
        if self.adaptive == None:
            # initialise scalar param, but don't register it
            self.logitalpha = theano.shared(
                value=numpy.array(_logit(numpy.sqrt(p / (1. - p)))).astype(theano.config.floatX),
                name='logitalpha'
            )
        elif self.adaptive == "layerwise":
            # initialise scalar param, allow updates
            self.logitalpha = theano.shared(
                value=numpy.array(_logit(numpy.sqrt(p / (1. - p)))).astype(theano.config.floatX),
                name='logitalpha'
            )
            self.add_param(self.logitalpha, ())
        elif self.adaptive == "elementwise":
            # initialise param for each activation passed
            self.logitalpha = theano.shared(
                value=numpy.array(
                    numpy.ones(self.input_shape[1:]) * _logit(numpy.sqrt(p / (1. - p)))
                ).astype(theano.config.floatX),
                name='logitalpha'
            )
            self.add_param(self.logitalpha, self.input_shape[1:])
        elif self.adaptive == "weightwise":
            # this will only work in the case of dropout type B
            thetashape = (self.input_shapes[1][1], self.input_shapes[0][1])
            self.logitalpha = theano.shared(
                value=numpy.array(
                    numpy.ones(thetashape) * _logit(numpy.sqrt(p / (1. - p)))
                ).astype(theano.config.floatX),
                name='logitalpha'
            )
            self.add_param(self.logitalpha, thetashape)
    '''

class VariationalDropoutTypeALayer(VariationalDropoutLayer, GaussianDropoutLayer):
    """
    Variational dropout layer, implementing correlated weight noise over the
    output of a layer. Adaptive version of Srivastava's Gaussian dropout.

    Inits:
        * p - initialisation of the parameters sampled for the noise
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, activation_probability=0.5, adaptive=None, nonlinearity=None,
                 **kwargs):
        VariationalDropoutLayer.__init__(self, incoming, activation_probability=activation_probability, adaptive=adaptive,
                                         nonlinearity=nonlinearity, **kwargs)

class VariationalDropoutTypeBLayer(FastDropoutLayer, VariationalDropoutLayer):
    """
    Variational dropout layer, implementing independent weight noise. Adaptive
    version of Wang's Gaussian dropout.

    Inits:
        * p - initialisation of the parameters sampled for the noise
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, activation_probability=0.5, adaptive=None, **kwargs):
        FastDropoutLayer.__init__(self, incoming, activation_probability=activation_probability, **kwargs)
        self.init_adaptive(adaptive, activation_probability=activation_probability)

#
#
#
#
#

class BetaBernoulliDropoutLayer(Layer):
    """Dropout layer with beta prior.

    Sets values to zero with probability activation_probability. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    activation_probability : float or scalar tensor
        The probability of setting a value to 1
    rescale : bool
        If true the input is rescaled with input / activation_probability when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / activation_probability when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, beta_prior_alpha, beta_prior_beta, rescale=True, **kwargs):
        super(BetaBernoulliDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        self.beta_prior_alpha = beta_prior_alpha;
        self.beta_prior_beta = beta_prior_beta;
        #self.activation_probability = activation_probability;

        self.rescale = rescale

    def update_prior(self, positive_samples, negative_samples):
        return;

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        retain_prob = numpy.random.beta(self.beta_prior_alpha, self.beta_prior_beta);

        if deterministic or numpy.all(retain_prob == 1):
            return input
        else:
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            mask = get_filter(input_shape, retain_prob, rng=RandomStreams())
            #mask = self.get_output(input_shape, retain_prob, rng=RandomStreams());

            if kwargs.get("mask_only", False):
                return mask;

            return input * mask;

    def get_output(self, input_shape, retain_prob, rng=RandomStreams()):
        return get_filter(input_shape, retain_prob, rng);

#
#
#
#
#

'''
def check_p_backup(p):
    """
    Thanks to our logit parameterisation we can't accept p of greater than or
    equal to 0.5 (or we get inf logitalphas). So we'll just warn the user and
    scale it down slightly.
    """
    if p == 0.5:
        warnings.warn("Cannot set p to exactly 0.5, limits are: 0 < p < 0.5."
                " Setting to 0.4999", RuntimeWarning)
        return 0.4999
    elif p > 0.5:
        warnings.warn("Cannot set p to greater than 0.5, limits are: "
                "0 < p < 0.5. Setting to 0.4999", RuntimeWarning)
        return 0.4999
    elif p <= 0.0:
        warnings.warn("Cannot set p to less than or equal to 0.0, limits are: "
                "0 < p < 0.5. Setting to 0.0001", RuntimeWarning)
        return 0.0001
    else:
        return p
'''

class GeneralizedDropoutLayerBackup(Layer):
    """Generalized Dropout layer

    Sets values to zero with probability activation_probability. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    activation_probability : float or scalar tensor
        The probability of setting a value to 1
    rescale : bool
        If true the input is rescaled with input / activation_probability when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / activation_probability when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, activation_probability, rescale=True, **kwargs):
        super(GeneralizedDropoutLayerBackup, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        self.activation_probability = activation_probability;

        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or numpy.all(self.activation_probability == 1):
            return input
        else:
            retain_prob = self.activation_probability
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * get_filter(input_shape, retain_prob, rng=RandomStreams())
