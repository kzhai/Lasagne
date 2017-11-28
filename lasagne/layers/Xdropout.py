import numpy
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import Layer, BernoulliDropoutLayerBackup
from .. import init
from ..random import get_rng

__all__ = [
	"BernoulliDropoutLayer",
	"DynamicDropoutLayer",
	#
	"BernoulliDropoutLayerHan",
]


class AdaptiveDropoutLayerBackup(Layer):
	"""Adaptive Dropout layer
	"""

	def __init__(self, incoming, activation_probability=0.5, rescale=True, shared_axes=(),
	             num_leading_axes=1, **kwargs):
		super(AdaptiveDropoutLayerBackup, self).__init__(incoming, **kwargs)
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
		# self.activation_probability = activation_probability

		self.activation_probability = self.add_param(activation_probability, self.input_shape[self.num_leading_axes:],
		                                             name="adaptable.r", trainable=False, regularizable=False,
		                                             adaptable=True)
		# self.activation_probability = theano.shared(value=activation_probability, )
		self.shared_axes = tuple(shared_axes)

	def _set_r(self, activation_probability=init.Uniform(range=(0, 1))):
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

	def _set_r(self, activation_probability=init.Uniform(range=(0, 1))):
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


'''
class PrunableAdaptiveDropoutLayer(AdaptiveDropoutLayer):
	"""Prunable Dropout layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)),
				 num_leading_axes=1, shared_axes=(), **kwargs):
		super(PrunableAdaptiveDropoutLayer, self).__init__(incoming, activation_probability, num_leading_axes,
														   shared_axes, **kwargs)

	def prune_activation_probability(self, input_indices_to_keep):
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		activation_probability = self.activation_probability.eval()[input_indices_to_keep]
		old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability
'''


class DynamicDropoutLayer(BernoulliDropoutLayer):
	"""Elastic adaptive Dropout layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)), num_leading_axes=1, shared_axes=(),
	             **kwargs):
		super(DynamicDropoutLayer, self).__init__(incoming, activation_probability, num_leading_axes,
		                                          shared_axes, **kwargs)

	def find_neuron_indices_to_prune(self, prune_threshold=1e-3):
		activation_probability = self.activation_probability.eval()
		neuron_indices_to_prune = numpy.argwhere(activation_probability < prune_threshold).flatten()
		neuron_indices_to_keep = numpy.setdiff1d(numpy.arange(0, len(activation_probability)), neuron_indices_to_prune)

		return neuron_indices_to_prune, neuron_indices_to_keep

	def prune_activation_probability(self, input_indices_to_keep):
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		activation_probability = self.activation_probability.eval()[input_indices_to_keep]
		old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability

	def find_neuron_indices_to_split(self, split_threshold=1 - 1e-3):
		activation_probability = self.activation_probability.eval()
		neuron_indices_to_split = numpy.argwhere(activation_probability > split_threshold).flatten()
		neuron_indices_to_keep = numpy.setdiff1d(numpy.arange(0, len(activation_probability)), neuron_indices_to_split)

		return neuron_indices_to_split, neuron_indices_to_keep

	def split_activation_probability(self, input_indices_to_split):
		old_size = int(numpy.prod(self.input_shape[self.num_leading_axes:]))
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == old_size + len(input_indices_to_split)

		activation_probability = self.activation_probability.eval()
		activation_probability = numpy.clip(activation_probability, 0, 1)
		split_ratio = numpy.random.random(len(input_indices_to_split))
		old_activation_probability = activation_probability[input_indices_to_split]
		activation_probability_split = activation_probability[input_indices_to_split] * split_ratio
		activation_probability[input_indices_to_split] *= 1 - split_ratio
		activation_probability = numpy.hstack((activation_probability, activation_probability_split))

		old_activation_probability = self._set_r(activation_probability)
		return old_activation_probability


class BernoulliDropoutLayerHan(BernoulliDropoutLayer):
	"""Prunable Dropout layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)),
	             num_leading_axes=1, shared_axes=(), **kwargs):
		super(BernoulliDropoutLayerHan, self).__init__(incoming, activation_probability, num_leading_axes,
		                                               shared_axes, **kwargs)

	def prune_activation_probability(self, input_indices_to_keep):
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		activation_probability = self.activation_probability.eval()[input_indices_to_keep]
		old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability

	def decay_activation_probability(self, decay_weight):
		activation_probability = self.activation_probability.eval() * decay_weight
		old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability


#
#
#
#
#

class GenericDropoutLayer(Layer):
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

	def __init__(self, incoming, activation_probability, num_leading_axes=1, **kwargs):
		super(GenericDropoutLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))

		num_inputs = int(numpy.prod(self.input_shape[num_leading_axes:]))
		if isinstance(activation_probability, numpy.ndarray):
			assert activation_probability.shape == (num_inputs,)
		self.activation_probability = self.add_param(activation_probability, (num_inputs,), name="r",
		                                             trainable=False, regularizable=False, adaptable=True)

	def get_output_for(self, input, deterministic=False, **kwargs):
		"""
		Parameters
		----------
		input : tensor
			output from the previous layer
		deterministic : bool
			If true dropout and scaling is disabled, see notes
		"""
		retain_prob = self.activation_probability.eval()
		if deterministic or numpy.all(retain_prob == 1):
			return input
		else:
			# retain_prob = self.activation_probability.eval()
			# if self.rescale:
			# input /= retain_prob

			# use nonsymbolic shape for dropout mask if possible
			input_shape = self.input_shape
			if any(s is None for s in input_shape):
				input_shape = input.shape

			return input * get_filter(input_shape, retain_prob, rng=RandomStreams())


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

		self.beta_prior_alpha = beta_prior_alpha
		self.beta_prior_beta = beta_prior_beta
		# self.activation_probability = activation_probability

		self.rescale = rescale

	def update_prior(self, positive_samples, negative_samples):
		return

	def get_output_for(self, input, deterministic=False, **kwargs):
		"""
		Parameters
		----------
		input : tensor
			output from the previous layer
		deterministic : bool
			If true dropout and scaling is disabled, see notes
		"""
		retain_prob = numpy.random.beta(self.beta_prior_alpha, self.beta_prior_beta)

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
			# mask = self.get_output(input_shape, retain_prob, rng=RandomStreams())

			if kwargs.get("mask_only", False):
				return mask

			return input * mask

	def get_output(self, input_shape, retain_prob, rng=RandomStreams()):
		return get_filter(input_shape, retain_prob, rng)
