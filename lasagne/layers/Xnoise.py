import sys
import warnings

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from lasagne.layers import Layer, MergeLayer
from lasagne.layers import get_all_layers
from lasagne import init, layers, nonlinearities
from lasagne import Xinit
from lasagne.random import get_rng

__all__ = [
	"sample_activation_probability",
	#
	"BernoulliDropoutLayer",
	#
	"GaussianDropoutSrivastavaLayer",
	"GaussianDropoutWangLayer",
	#
	"VariationalDropoutSrivastavaLayer",
	"VariationalDropoutWangLayer",
	#
	"VariationalDropoutLayer",
	"SparseVariationalDropoutLayer",
	#
	"AdaptiveDropoutLayer",
	#
	"BernoulliDropoutLayerHan",
]


def sample_activation_probability(input_dimensions, activation_style, activation_parameter):
	if activation_style == "uniform":
		activation_probability = numpy.random.random(size=input_dimensions)
	elif activation_style == "bernoulli":
		activation_probability = numpy.zeros(input_dimensions) + activation_parameter
	elif activation_style == "beta_bernoulli":
		shape_alpha, shape_beta = activation_parameter
		activation_probability = numpy.random.beta(shape_alpha, shape_beta, size=input_dimensions)
	elif activation_style == "reciprocal_beta_bernoulli":
		shape_alpha, shape_beta = activation_parameter
		input_number_of_neurons = numpy.prod(input_dimensions)
		activation_probability = numpy.zeros(input_number_of_neurons)
		ranked_shape_alpha = shape_alpha / numpy.arange(1, input_number_of_neurons + 1)
		for index in range(input_number_of_neurons):
			activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta)
		activation_probability = numpy.reshape(activation_probability, input_dimensions)
	elif activation_style == "reverse_reciprocal_beta_bernoulli":
		shape_alpha, shape_beta = activation_parameter
		ranked_shape_alpha = shape_alpha / numpy.arange(1, input_dimensions + 1)[::-1]
		input_number_of_neurons = numpy.prod(input_dimensions)
		activation_probability = numpy.zeros(input_number_of_neurons)
		for index in range(input_number_of_neurons):
			activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta)
		activation_probability = numpy.reshape(activation_probability, input_dimensions)
	elif activation_style == "mixed_beta_bernoulli":
		beta_mean, shape_beta = activation_parameter
		scale = beta_mean / (1. - beta_mean)
		input_number_of_neurons = numpy.prod(input_dimensions)
		activation_probability = numpy.zeros(input_number_of_neurons)
		for index in range(input_number_of_neurons):
			rank = index + 1
			activation_probability[index] = numpy.random.beta(rank * scale / shape_beta, rank / shape_beta)
		activation_probability = numpy.reshape(activation_probability, input_dimensions)
	elif activation_style == "geometric":
		input_number_of_neurons = numpy.prod(input_dimensions)
		activation_probability = numpy.zeros(input_number_of_neurons)
		for index in range(input_number_of_neurons):
			rank = index + 1
			activation_probability[index] = (activation_parameter - 1) / numpy.log(activation_parameter) * (
					activation_parameter ** rank)
		activation_probability = numpy.clip(activation_probability, 0., 1.)
		activation_probability = numpy.reshape(activation_probability, input_dimensions)
	elif activation_style == "reciprocal":
		activation_probability = activation_parameter / numpy.arange(1, input_dimensions + 1)
		activation_probability = numpy.clip(activation_probability, 0., 1.)
	elif activation_style == "exponential":
		activation_probability = activation_parameter / numpy.arange(1, input_dimensions + 1)
		activation_probability = numpy.clip(activation_probability, 0., 1.)
	else:
		sys.stderr.serialize("error: unrecognized configuration...\n")
		sys.exit()

	return activation_probability.astype(theano.config.floatX)


class BernoulliDropoutLayer(Layer):
	"""Adaptive Dropout layer
	"""

	def __init__(self, incoming, activation_probability=0.5, shared_axes=(), num_leading_axes=1, **kwargs):
		super(BernoulliDropoutLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))

		if num_leading_axes >= len(self.input_shape):
			raise ValueError(
				"Got num_leading_axes=%d for a %d-dimensional input, "
				"leaving no trailing axes for the dot product." %
				(num_leading_axes, len(self.input_shape)))
		elif num_leading_axes < -len(self.input_shape):
			raise ValueError(
				"Got num_leading_axes=%d for a %d-dimensional input, "
				"requesting more trailing axes than there are input "
				"dimensions." % (num_leading_axes, len(self.input_shape)))
		self.num_leading_axes = num_leading_axes

		self.activation_probability = self.add_param(activation_probability, self.input_shape[self.num_leading_axes:],
		                                             name="r", trainable=False, regularizable=False)

		self.shared_axes = tuple(shared_axes)

	def _set_activation_probability(self, activation_probability=init.Uniform(range=(0, 1))):
		old_activation_probability = self.activation_probability.eval()
		self.params.pop(self.activation_probability)

		if isinstance(activation_probability, init.Initializer):
			self.activation_probability = self.add_param(activation_probability,
			                                             self.input_shape[self.num_leading_axes:], name="r",
			                                             trainable=False, regularizable=False)
		elif isinstance(activation_probability, numpy.ndarray):
			self.activation_probability = self.add_param(activation_probability, activation_probability.shape, name="r",
			                                             trainable=False, regularizable=False)
		else:
			raise ValueError("Unrecognized parameter type %s." % type(activation_probability))

		return old_activation_probability

	def get_output_for(self, input, deterministic=False, **kwargs):
		if deterministic:
			# return input * self.activation_probability
			return T.mul(input, T.clip(self.activation_probability, 0, 1))
		else:
			retain_prob = self.activation_probability.eval()
			retain_prob = numpy.clip(retain_prob, 0, 1)

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


def get_filter(input_shape, retain_probability, rng=RandomStreams()):
	filter = rng.binomial(size=input_shape, n=1, p=retain_probability, dtype=theano.config.floatX)

	return filter


def _logit(x):
	"""
	Logit function in Numpy. Useful for parameterizing alpha.
	"""
	return numpy.log(x / (1. - x)).astype(theano.config.floatX)


def _sigmoid(x):
	"""
	Logit function in Numpy. Useful for parameterizing alpha.
	"""
	return 1. / (1 + numpy.exp(-x)).astype(theano.config.floatX)


class GaussianDropoutSrivastavaLayer(Layer):
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

	def __init__(self, incoming, activation_probability=0.5, shared_axes=(), num_leading_axes=1, **kwargs):
		super(GaussianDropoutSrivastavaLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))

		self.shared_axes = tuple(shared_axes)

		activation_probability = _validate_activation_probability_for_logit_parameterization(activation_probability)
		# self.logit_sigma = _logit(numpy.sqrt((1 - activation_probability) / activation_probability))
		logit_sigma = _logit(numpy.sqrt((1 - activation_probability) / activation_probability))
		self.logit_sigma = self.add_param(logit_sigma, logit_sigma.shape, name="logit_sigma", trainable=False,
		                                  regularizable=False)

	def get_output_for(self, input, deterministic=False, **kwargs):
		"""
		Parameters
		----------
		input : tensor
		output from the previous layer
		deterministic : bool
		If true noise is disabled, see notes
		"""
		self.sigma = T.nnet.sigmoid(self.logit_sigma)
		if deterministic or numpy.all(self.sigma.eval() == 0):
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
			perturbation = 1 + self.sigma * self._srng.normal(input.shape, avg=0.0, std=1.)

			if self.shared_axes:
				bcast = tuple(bool(s == 1) for s in perturbation_shape)
				perturbation = T.patternbroadcast(perturbation, bcast)
			return input * perturbation


class GaussianDropoutWangLayer(MergeLayer):
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

		activation_probability = _validate_activation_probability_for_logit_parameterization(activation_probability)
		logit_sigma = _logit(numpy.sqrt((1 - activation_probability) / activation_probability))
		self.logit_sigma = self.add_param(logit_sigma, logit_sigma.shape, name="logit_sigma", trainable=False,
		                                  regularizable=False)

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
			inputs[1] = inputs[1].flatten(2)
		self.sigma = T.nnet.sigmoid(self.logit_sigma)
		mu_z = T.dot(inputs[1], self.theta) + self.b.dimshuffle('x', 0)
		if deterministic or numpy.all(self.sigma.eval() == 0):
			return self.nonlinearity(mu_z)
		else:
			# sample from the Gaussian that dropout would produce
			sigma_z = T.sqrt(T.dot(T.square(inputs[1]),
			                       self.sigma * T.square(self.theta)))
			randn = self._srng.normal(size=inputs[0].shape, avg=0.0, std=1.)
			return self.nonlinearity(mu_z + sigma_z * randn)


class VariationalDropoutBaseLayer(Layer):
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

	def __init__(self, incoming, activation_probability=0.5, adaptive="elementwise", **kwargs):
		super(VariationalDropoutBaseLayer, self).__init__(incoming, **kwargs)

		'''
		num_inputs = int(numpy.prod(self.input_shape[num_leading_axes:]))
		if isinstance(activation_probability, numpy.ndarray):
			assert activation_probability.shape == (num_inputs,)
		'''

		self.init_adaptive(activation_probability, adaptive)

	def init_adaptive(self, activation_probability, adaptive):
		"""
		Initialises adaptive parameters.
		"""
		if not hasattr(self, 'input_shape'):
			self.input_shape = self.input_shapes[0]

		self.adaptive = adaptive
		activation_probability = _validate_activation_probability_for_logit_parameterization(activation_probability)

		# init based on adaptive options:
		'''
		if self.adaptive == None:
			# initialise scalar param, but don't register it
			logit_sigma = _logit(numpy.sqrt((1 - activation_probability) / activation_probability))
			self.logit_sigma = self.add_param(logit_sigma, (), name="variational.dropout.logit_sigma",
											  trainable=False, regularizable=False)
		'''
		if self.adaptive == "layerwise":
			# initialise scalar param, allow updates
			logit_sigma = _logit(numpy.sqrt((1 - activation_probability) / activation_probability))
			self.logit_sigma = self.add_param(logit_sigma, (), name="variational.dropout.logit_sigma",
			                                  trainable=True, regularizable=False)
		elif self.adaptive == "elementwise":
			# initialise param for each activation passed
			logit_sigma = _logit(numpy.ones(self.input_shape[1:]) * numpy.sqrt((1 - activation_probability) /
			                                                                   activation_probability))
			self.logit_sigma = self.add_param(logit_sigma, self.input_shape[1:],
			                                  name="variational.dropout.logit_sigma", trainable=True,
			                                  regularizable=False)
		elif self.adaptive == "weightwise":
			# this will only work in the case of dropout type B
			thetashape = (self.input_shapes[1][1], self.input_shapes[0][1])
			logit_sigma = _logit(numpy.ones(thetashape) * numpy.sqrt((1 - activation_probability) /
			                                                         activation_probability))
			self.logit_sigma = self.add_param(logit_sigma, thetashape, name="variational.dropout.logit_sigma",
			                                  trainable=True, regularizable=False)
		else:
			sys.stderr.serialize("error: unrecognized configuration...\n")
			sys.exit()


class VariationalDropoutSrivastavaLayer(VariationalDropoutBaseLayer, GaussianDropoutSrivastavaLayer):
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

	def __init__(self, incoming, activation_probability=0.5, adaptive="elementwise",
	             **kwargs):
		VariationalDropoutBaseLayer.__init__(self, incoming, activation_probability=activation_probability,
		                                     adaptive=adaptive, **kwargs)


class VariationalDropoutWangLayer(GaussianDropoutWangLayer, VariationalDropoutBaseLayer):
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

	def __init__(self, incoming, activation_probability=0.5, adaptive="elementwise", **kwargs):
		GaussianDropoutWangLayer.__init__(self, incoming, activation_probability, **kwargs)
		self.init_adaptive(activation_probability, adaptive)


def _validate_activation_probability_for_logit_parameterization(activation_probability, clip_margin=1e-6):
	"""
	Thanks to our logit parameterisation we can't accept p of smaller or equal
	to 0.5 nor greater or equal to 1. So we'll just warn the user and
	scale it down slightly.
	"""

	if numpy.any(activation_probability <= 0.5) or numpy.any(activation_probability >= 1.0):
		warnings.warn("Clipping p to the interval of (0.5, 1.0).", RuntimeWarning)
		return numpy.clip(activation_probability, 0.5 + clip_margin, 1 - clip_margin)
	return numpy.asarray(activation_probability).astype(theano.config.floatX)


VariationalDropoutLayer = VariationalDropoutSrivastavaLayer
SparseVariationalDropoutLayer = VariationalDropoutSrivastavaLayer


class AdaptiveDropoutLayer(BernoulliDropoutLayer):
	"""Adaptive Dropout layer
	"""

	def __init__(self, incoming, activation_probability=0.5, shared_axes=(), num_leading_axes=1, **kwargs):
		super(AdaptiveDropoutLayer, self).__init__(incoming=incoming, activation_probability=activation_probability,
		                                           shared_axes=shared_axes, num_leading_axes=num_leading_axes, **kwargs)
		self.params.pop(self.activation_probability)

		self.activation_probability = self.add_param(activation_probability, self.input_shape[self.num_leading_axes:],
		                                             name="adaptable.r", trainable=False, regularizable=False,
		                                             adaptable=True)

	def _set_activation_probability(self, activation_probability=init.Uniform(range=(0, 1))):
		old_activation_probability = self.activation_probability.eval()
		self.params.pop(self.activation_probability)

		if isinstance(activation_probability, init.Initializer):
			self.activation_probability = self.add_param(activation_probability,
			                                             self.input_shape[self.num_leading_axes:],
			                                             name="adaptable.r", trainable=False,
			                                             regularizable=False, adaptable=True)
		elif isinstance(activation_probability, numpy.ndarray):
			self.activation_probability = self.add_param(activation_probability,
			                                             activation_probability.shape,
			                                             name="adaptable.r", trainable=False,
			                                             regularizable=False, adaptable=True)
		else:
			raise ValueError("Unrecognized parameter type %s." % type(activation_probability))

		return old_activation_probability


class BernoulliDropoutLayerHan(BernoulliDropoutLayer):
	"""Bernoulli Dropout Layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)), shared_axes=(), num_leading_axes=1,
	             **kwargs):
		super(BernoulliDropoutLayerHan, self).__init__(incoming, activation_probability, shared_axes, num_leading_axes,
		                                               **kwargs)

	def prune_activation_probability(self, input_indices_to_keep):
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		activation_probability = self.activation_probability.eval()[input_indices_to_keep]
		old_activation_probability = self._set_activation_probability(activation_probability)

		return old_activation_probability

	def decay_activation_probability(self, decay_weight):
		old_activation_probability = self.activation_probability.eval()
		activation_probability = old_activation_probability * decay_weight
		self.activation_probability.set_value(activation_probability)
		# old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability
