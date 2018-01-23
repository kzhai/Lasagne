import logging

import numpy
import theano.tensor as T

from . import Layer, DenseLayer
from .. import init
from .. import nonlinearities

logger = logging.getLogger(__name__)

__all__ = [
	"AdaptableDenseLayer",
	# "PrunableDenseLayer",
	# "SplitableDenseLayer",
	"DynamicDenseLayer",
	#
	#
	#
	"ObstructedDenseLayer",
	#
	#
	#
	"MaskedDenseLayerHan",
	"MaskedDenseLayerGuo",
]


class AdaptableDenseLayer(DenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(AdaptableDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes, **kwargs)

	def _set_W(self, W=init.GlorotUniform()):
		old_W = self.W.eval()
		self.params.pop(self.W)

		if isinstance(W, init.Initializer):
			raise ValueError("Reset %s parameter without alternating its dimensionality.\n" % self +
			                 "Consider switch to 'set_subtensor' in this case for efficiency.")
			num_inputs = int(numpy.prod(self.input_shape[self.num_leading_axes:]))
			self.W = self.add_param(W, (num_inputs, self.num_units), name="W")
		elif isinstance(W, numpy.ndarray):
			self.W = self.add_param(W, W.shape, name="W")
		else:
			raise ValueError("Unrecognized parameter type %s." % type(W))

		return old_W

	def _set_b(self, b=init.Constant(0.)):
		old_b = self.b.eval()
		self.params.pop(self.b)

		if b is None:
			raise ValueError("Reset %s parameter without alternating its dimensionality.\n" % self +
			                 "Consider switch to 'set_subtensor' in this case for efficiency.")
			self.b = None
		elif isinstance(b, init.Initializer):
			raise ValueError("Reset %s parameter without alternating its dimensionality.\n" % self +
			                 "Consider switch to 'set_subtensor' in this case for efficiency.")
			self.b = self.add_param(b, (self.num_units,), name="b", regularizable=False)
		elif isinstance(b, numpy.ndarray):
			self.b = self.add_param(b, b.shape, name="b", regularizable=False)
		else:
			raise ValueError("Unrecognized parameter type %s." % type(b))

		return old_b


class DynamicDenseLayer(AdaptableDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(DynamicDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity,
		                                        num_leading_axes, **kwargs)

	def prune_input(self, input_indices_to_keep):
		# print('prune input of layer %s from %d to %d' % (self, int(np.prod(self.input_shape[self.num_leading_axes:])), len(input_indices_to_keep)))

		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		W = self.W.eval()[input_indices_to_keep, :]
		old_W = self._set_W(W)

		return old_W

	def prune_output(self, output_indices_to_keep):
		# print('prune output of layer %s from size %d to %d' % (self, self.num_units, len(output_indices_to_keep)))

		W = self.W.eval()[:, output_indices_to_keep]
		b = self.b.eval()[output_indices_to_keep]

		self.num_units = len(output_indices_to_keep)

		old_W = self._set_W(W)
		old_b = self._set_b(b)

		return old_W, old_b

	def split_input(self, input_indices_to_split):
		self.input_shape = self.input_layer.output_shape
		# assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_split)

		W = self.W.eval()
		W = numpy.vstack((W, W[input_indices_to_split, :]))
		old_W = self._set_W(W)

		return old_W

	"""
	@deprecated
	"""

	def split_output_dropout(self, output_indices_to_split):
		W = self.W.eval()
		b = self.b.eval()

		W = numpy.hstack((W, W[:, output_indices_to_split]))
		b = numpy.hstack((b, b[output_indices_to_split]))

		self.num_units += len(output_indices_to_split)

		old_W = self._set_W(W)
		old_b = self._set_b(b)

		return old_W, old_b

	def split_output(self, output_indices_to_split, **kwargs):
		W = self.W.eval()
		b = self.b.eval()

		split_mode = kwargs.get("split_mode", "dense")
		if split_mode == "dense":
			W, b = split_W_and_b(W, b, output_indices_to_split)

			'''
			split_strategy = kwargs.get("split_strategy", "weightwise")
			if split_strategy == "layerwise":
				W_split_ratio = numpy.random.random()
				b_split_ratio = numpy.random.random()
			elif split_strategy == "neuronwise":
				W_split_ratio = numpy.random.random((1, len(output_indices_to_split)))
				b_split_ratio = numpy.random.random(len(output_indices_to_split))
			elif split_strategy == "weightwise":
				W_split_ratio = numpy.random.random(W[:, output_indices_to_split].shape)
				b_split_ratio = numpy.random.random(len(output_indices_to_split))
			else:
				raise ValueError("Unrecognized split strategy %s." % (split_strategy))

			W_split = W[:, output_indices_to_split] * W_split_ratio
			W[:, output_indices_to_split] *= 1 - W_split_ratio
			W = numpy.hstack((W, W_split))

			b_split = b[output_indices_to_split] * b_split_ratio
			b[output_indices_to_split] *= 1 - b_split_ratio
			b = numpy.hstack((b, b_split))
			'''
		elif split_mode == "dropout":
			W = numpy.hstack((W, W[:, output_indices_to_split]))
			b = numpy.hstack((b, b[output_indices_to_split]))
		else:
			raise ValueError("Unrecognized split model %s." % (split_mode))

		self.num_units += len(output_indices_to_split)

		old_W = self._set_W(W)
		old_b = self._set_b(b)

		return old_W, old_b

def split_W_and_b(W, b, indices_to_split, split_strategy="weightwise"):
	# split_strategy = kwargs.get("split_strategy", "weightwise")
	if split_strategy == "layerwise":
		W_split_ratio = numpy.random.random()
		b_split_ratio = numpy.random.random()
	elif split_strategy == "neuronwise":
		W_split_ratio = numpy.random.random((1, len(indices_to_split)))
		b_split_ratio = numpy.random.random(len(indices_to_split))
	elif split_strategy == "weightwise":
		W_split_ratio = numpy.random.random(W[:, indices_to_split].shape)
		b_split_ratio = numpy.random.random(len(indices_to_split))
	else:
		raise ValueError("Unrecognized split strategy %s." % (split_strategy))

	W_split = W[:, indices_to_split] * W_split_ratio
	W[:, indices_to_split] *= 1 - W_split_ratio

	b_split = b[indices_to_split] * b_split_ratio
	b[indices_to_split] *= 1 - b_split_ratio

	# W = numpy.hstack((W, W_split))
	# b = numpy.hstack((b, b_split))

	return numpy.hstack((W, W_split)), numpy.hstack((b, b_split))


#
#
#
#
#

class MaskedDenseLayer(DenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, mask=None, **kwargs):
		super(MaskedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes,
		                                       **kwargs)

		num_inputs = int(numpy.prod(self.input_shape[self.num_leading_axes:]))
		if mask == None:
			self.mask = numpy.ones((num_inputs, self.num_units))
		else:
			assert mask.shape == (num_inputs, self.num_units)
			self.mask = mask

	def adjust_mask(self):
		raise NotImplementedError("Not implemented in successor classes!")

	def get_output_for(self, input, **kwargs):
		num_leading_axes = self.num_leading_axes
		if num_leading_axes < 0:
			num_leading_axes += input.ndim
		if input.ndim > num_leading_axes + 1:
			# flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
			input = input.flatten(num_leading_axes + 1)

		activation = T.dot(input, self.mask * self.W)
		if self.b is not None:
			activation = activation + self.b
		return self.nonlinearity(activation)


class MaskedDenseLayerHan(MaskedDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, mask=None, **kwargs):
		super(MaskedDenseLayerHan, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes, mask,
		                                          **kwargs)

	def adjust_mask(self, threshold=1e-3):
		# threshold = numpy.min(numpy.max(numpy.abs(self.W.eval()), axis=0))
		# print("number of connections: %d" % np.sum(self.mask == 1))

		# self.mask = numpy.abs(self.W.eval())
		W = self.W.eval()
		self.mask *= (numpy.abs(W) > threshold)
		return self.mask


class MaskedDenseLayerGuo(MaskedDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, mask=None, **kwargs):
		super(MaskedDenseLayerGuo, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes, mask,
		                                          **kwargs)

	def adjust_mask(self, thresholds=(0, 1)):
		# threshold = numpy.min(numpy.max(numpy.abs(self.W.eval()), axis=0))
		# print("number of connections: %d" % np.sum(self.mask == 1))

		W = self.W.eval()
		self.mask *= (numpy.abs(W) > thresholds[0])
		self.mask *= (numpy.abs(W) < thresholds[1])
		self.mask += (numpy.abs(W) > thresholds[1])

		return self.mask


#
#
#
#
#

class ObstructedDenseLayer(Layer):
	def __init__(self, incoming, num_units, obstructed_incoming, num_obstructed_units, W=init.GlorotUniform(),
	             b=init.Constant(0.), nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		assert num_obstructed_units > 0 and num_obstructed_units < num_units

		# num_inputs = int(numpy.prod(self.input_shape[num_leading_axes:]))
		# self.obstructed_W = self.add_param(init.GlorotUniform(), (num_inputs, num_obstructed_units), name="obstructedW", trainable=False, adaptable=False)

		# self.num_obstructed_units = num_obstructed_units

		super(ObstructedDenseLayer, self).__init__(incoming, **kwargs)
		self.nonlinearity = (nonlinearities.identity if nonlinearity is None
		                     else nonlinearity)

		self.num_units = num_units

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

		if any(s is None for s in self.input_shape[num_leading_axes:]):
			raise ValueError(
				"A DenseLayer requires a fixed input shape (except for "
				"the leading axes). Got %r for num_leading_axes=%d." %
				(self.input_shape, self.num_leading_axes))
		num_inputs = int(numpy.prod(self.input_shape[num_leading_axes:]))

		self.W = self.add_param(W, (num_inputs, num_units - num_obstructed_units), name="W")
		if b is None:
			self.b = None
		else:
			self.b = self.add_param(b, (num_units - num_obstructed_units,), name="b",
			                        regularizable=False)

		self.obstructed_input_layer = obstructed_incoming

	def get_output_shape_for(self, input_shape):
		return input_shape[:self.num_leading_axes] + (self.num_units,)

	def get_output_for(self, input, axis=1, **kwargs):
		num_leading_axes = self.num_leading_axes
		if num_leading_axes < 0:
			num_leading_axes += input.ndim
		if input.ndim > num_leading_axes + 1:
			# flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
			input = input.flatten(num_leading_axes + 1)

		activation = T.dot(input, self.W)
		if self.b is not None:
			activation = activation + self.b

		return T.concatenate([self.obstructed_input_layer, self.nonlinearity(activation)], axis=axis)


class PrunableDenseLayer(AdaptableDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(PrunableDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes,
		                                         **kwargs)

	def prune_input(self, input_indices_to_keep):
		# print('prune input of layer %s from %d to %d' % (self, int(np.prod(self.input_shape[self.num_leading_axes:])), len(input_indices_to_keep)))

		self.input_shape = self.input_layer.output_shape
		assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_keep)

		W = self.W.eval()[input_indices_to_keep, :]
		old_W = self._set_W(W)

		return old_W

	def prune_output(self, output_indices_to_keep):
		# print('prune output of layer %s from size %d to %d' % (self, self.num_units, len(output_indices_to_keep)))

		W = self.W.eval()[:, output_indices_to_keep]
		b = self.b.eval()[output_indices_to_keep]

		self.num_units = len(output_indices_to_keep)

		old_W = self._set_W(W)
		old_b = self._set_b(b)

		return old_W, old_b


class SplitableDenseLayer(AdaptableDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(SplitableDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes,
		                                          **kwargs)

	def split_input(self, input_indices_to_split):
		self.input_shape = self.input_layer.output_shape
		# assert int(numpy.prod(self.input_shape[self.num_leading_axes:])) == len(input_indices_to_split)

		W = self.W.eval()
		W = numpy.vstack((W, W[input_indices_to_split, :]))
		old_W = self._set_W(W)

		return old_W

	def split_output(self, output_indices_to_split):
		W = self.W.eval()
		W = numpy.hstack((W, W[:, output_indices_to_split]))
		b = self.b.eval()
		b = numpy.hstack((b, b[output_indices_to_split]))

		self.num_units += len(output_indices_to_split)

		old_W = self._set_W(W)
		old_b = self._set_b(b)

		return old_W, old_b


if __name__ == '__main__':
	for i in range(100):
		x = numpy.random.random((2, 5))

		W_1 = numpy.random.random((5, 10))
		b_1 = numpy.random.random(10)

		W_2 = numpy.random.random((10, 20))
		b_2 = numpy.random.random(20)

		old_output_1 = numpy.dot(x, W_1) + b_1
		old_output_2 = numpy.dot(old_output_1, W_2) + b_2
		# print(old_output_2)

		input_indices_to_split = range(7)

		'''
		split_ratio = numpy.random.random((1, len(input_indices_to_split)))
		W_1_split = W_1[:, input_indices_to_split] * split_ratio
		W_1[:, input_indices_to_split] *= 1 - split_ratio
		W_1 = numpy.hstack((W_1, W_1_split))
		'''

		W_1, b_1 = split_W_and_b(W_1, b_1, input_indices_to_split)

		'''
		split_ratio = numpy.random.random(W_1[:, input_indices_to_split].shape)
		W_1_split = W_1[:, input_indices_to_split] * split_ratio
		W_1[:, input_indices_to_split] *= 1 - split_ratio
		W_1 = numpy.hstack((W_1, W_1_split))

		split_ratio = numpy.random.random(len(input_indices_to_split))
		b_1_split = b_1[input_indices_to_split] * split_ratio
		b_1[input_indices_to_split] *= 1 - split_ratio
		b_1 = numpy.hstack((b_1, b_1_split))
		'''

		W_2 = numpy.vstack((W_2, W_2[input_indices_to_split, :]))

		new_output_1 = numpy.dot(x, W_1) + b_1
		new_output_2 = numpy.dot(new_output_1, W_2) + b_2

		assert numpy.allclose(old_output_2, new_output_2), i
