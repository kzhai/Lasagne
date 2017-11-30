import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import BernoulliDropoutLayer
from .. import init
from ..random import get_rng

__all__ = [
	"AdaptiveDropoutLayer",
	"DynamicDropoutLayer",
	#
	"BernoulliDropoutLayerHan",
]


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


class DynamicDropoutLayer(AdaptiveDropoutLayer):
	"""Elastic adaptive Dropout layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)), num_leading_axes=1, shared_axes=(),
	             **kwargs):
		super(DynamicDropoutLayer, self).__init__(incoming=incoming, activation_probability=activation_probability,
		                                          num_leading_axes=num_leading_axes, shared_axes=shared_axes, **kwargs)

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
		old_activation_probability = self.activation_probability.eval()
		activation_probability = old_activation_probability * decay_weight
		self.activation_probability.set_value(activation_probability)
		# old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability


class BernoulliDropoutLayerHanBackup(AdaptiveDropoutLayer):
	"""Prunable Dropout layer
	"""

	def __init__(self, incoming, activation_probability=init.Uniform(range=(0, 1)),
	             num_leading_axes=1, shared_axes=(), **kwargs):
		super(BernoulliDropoutLayerHanBackup, self).__init__(incoming, activation_probability, num_leading_axes,
		                                                     shared_axes, **kwargs)

	def prune_activation_probability(self, input_indices_to_keep):
		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		activation_probability = self.activation_probability.eval()[input_indices_to_keep]
		old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability

	def decay_activation_probability(self, decay_weight):
		old_activation_probability = self.activation_probability.eval()
		activation_probability = old_activation_probability * decay_weight
		self.activation_probability.set_value(activation_probability)
		# old_activation_probability = self._set_r(activation_probability)

		return old_activation_probability
