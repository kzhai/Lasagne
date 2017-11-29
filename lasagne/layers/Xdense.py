import logging

import numpy
import theano.tensor as T

from . import Layer, DenseLayer
from .. import init
from .. import nonlinearities

logger = logging.getLogger(__name__)

__all__ = [
	"AdaptableDenseLayer",
	"PrunableDenseLayer",
	"SplitableDenseLayer",
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
		self.params.pop(self.W);

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
		W = numpy.vstack((W, W[input_indices_to_split, :]));
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


class DynamicDenseLayer(PrunableDenseLayer, SplitableDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(DynamicDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity,
		                                        num_leading_axes, **kwargs)

	'''
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
		W = numpy.vstack((W, W[input_indices_to_split, :]));
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
	'''


#
#
#
#
#



class DenseLayerHanBackup(PrunableDenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, num_leading_axes=1, **kwargs):
		super(DenseLayerHanBackup, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes,
		                                          **kwargs)
		num_inputs = int(numpy.prod(self.input_shape[self.num_leading_axes:]))
		self.mask = numpy.ones((num_inputs, self.num_units))

	def adjust_mask(self, threshold=1e-3):
		# threshold = numpy.min(numpy.max(numpy.abs(self.W.eval()), axis=0))
		# print("number of connections: %d" % np.sum(self.mask == 1))

		self.mask *= (numpy.abs(self.W.eval()) > threshold)
		return self.find_neuron_indices_to_prune()

	def find_neuron_indices_to_prune(self):
		number_of_active_synapses_per_neuron = numpy.sum(self.mask, axis=0)

		neuron_indices_to_prune = numpy.argwhere(number_of_active_synapses_per_neuron <= 0).flatten()
		neuron_indices_to_keep = numpy.setdiff1d(numpy.arange(0, len(self.mask)), neuron_indices_to_prune)
		return neuron_indices_to_prune, neuron_indices_to_keep

	def prune_input(self, input_indices_to_keep):
		old_W = super(DenseLayerHanBackup, self).prune_input(input_indices_to_keep)
		old_mask = numpy.copy(self.mask)
		self.mask = self.mask[input_indices_to_keep, :]

		return old_W, old_mask

	def prune_output(self, output_indices_to_keep):
		old_W, old_b = super(DenseLayerHanBackup, self).prune_output(output_indices_to_keep)
		old_mask = numpy.copy(self.mask)
		self.mask = self.mask[:, output_indices_to_keep]

		return old_W, old_b, old_mask

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

		#self.mask = numpy.abs(self.W.eval())
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

class BernoulliDropoutDenseLayer(DenseLayer):
	def __init__(self, incoming, num_units, activation_probability=0.5, rescale=True, W=init.GlorotNormal(),
	             b=init.Constant(0.), nonlinearity=nonlinearities.rectify, shared_axes=(), num_leading_axes=1,
	             **kwargs):
		super(BernoulliDropoutDenseLayer, self).__init__(incoming, num_units, W=W, b=b, nonlinearity=nonlinearity,
		                                                 num_leading_axes=num_leading_axes, **kwargs)
		# self.activation_probability = activation_probability
		self.activation_probability = self.add_param(activation_probability, self.input_shape[num_leading_axes:],
		                                             name="r", trainable=False, regularizable=False)

		self.rescale = rescale
		self.shared_axes = tuple(shared_axes)

		#
		#
		#
		#
		#

		self.reg = True
		self.num_updates = 0

	def get_output_for(self, input, deterministic=False, **kwargs):
		retain_prob = self.activation_probability.eval()
		if not (deterministic or numpy.all(retain_prob == 1)):
			# Using theano constant to prevent upcasting
			if self.rescale:
				T.true_div(input, self.activation_probability)
			# input /= retain_prob

			# use nonsymbolic shape for dropout mask if possible
			mask_shape = self.input_shape
			if any(s is None for s in mask_shape):
				mask_shape = input.shape

			# apply dropout, respecting shared axes
			if self.shared_axes:
				shared_axes = tuple(a if a >= 0 else a + input.ndim for a in self.shared_axes)
				mask_shape = tuple(1 if a in shared_axes else s for a, s in enumerate(mask_shape))
			mask = self._srng.binomial(mask_shape, p=retain_prob, dtype=input.dtype)
			if self.shared_axes:
				bcast = tuple(bool(s == 1) for s in mask_shape)
				mask = T.patternbroadcast(mask, bcast)
			input *= mask

		return self.nonlinearity(T.dot(input, self.W) + self.b)

	'''
	def eval_reg(self, **kwargs):
		return 0

	def get_ard(self, **kwargs):
		return None

	def get_reg(self):
		return str(self.activation_probability / (1 - self.activation_probability))
	'''


class DenseVarDropOutARD(DenseLayer):
	def __init__(self, incoming, num_units, activation_probability=0.5, rescale=True, W=init.GlorotNormal(),
	             b=init.Constant(0.), nonlinearity=nonlinearities.rectify, ard_init=-10, shared_axes=(),
	             num_leading_axes=1, **kwargs):
		super(DenseVarDropOutARD, self).__init__(incoming, num_units, W=W, b=b, nonlinearity=nonlinearity,
		                                         num_leading_axes=num_leading_axes, **kwargs)

		self.rescale = rescale
		self.shared_axes = tuple(shared_axes)

		#
		#
		#
		#
		#

		self.reg = True
		self.log_sigma2 = self.add_param(Constant(ard_init), (self.num_inputs, self.num_units), name="ls2")

	@staticmethod
	def clip(mtx, to=8):
		mtx = T.switch(T.le(mtx, -to), -to, mtx)
		mtx = T.switch(T.ge(mtx, to), to, mtx)

		return mtx

	def get_output_for(self, input, deterministic=False, train_clip=False, thresh=3, **kwargs):
		log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
		clip_mask = T.ge(log_alpha, thresh)

		if deterministic:
			activation = T.dot(input, T.switch(clip_mask, 0, self.W))
		else:
			W = self.W
			if train_clip:
				W = T.switch(clip_mask, 0, self.W)
			mu = T.dot(input, W)
			si = T.sqrt(T.dot(input * input, T.exp(log_alpha) * self.W * self.W) + 1e-8)
			activation = mu + self._srng.normal(mu.shape, avg=0, std=1) * si
		return self.nonlinearity(activation + self.b)

	def kl_divergence_approximation(self, **kwargs):
		k1, k2, k3 = 0.63576, 1.8732, 1.48695
		C = -k1
		log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
		mdkl = k1 * T.nnet.sigmoid(k2 + k3 * log_alpha) - 0.5 * T.log1p(T.exp(-log_alpha)) + C
		return -T.sum(mdkl)

	#
	#
	#
	#
	#

	def get_ard(self, thresh=3, **kwargs):
		log_alpha = self.log_sigma2.get_value() - 2 * numpy.log(numpy.abs(self.W.get_value()))
		return '%.4f' % (numpy.sum(log_alpha > thresh) * 1.0 / log_alpha.size)

	def get_reg(self):
		log_alpha = self.log_sigma2.get_value() - 2 * numpy.log(numpy.abs(self.W.get_value()))
		return '%.1f, %.1f' % (log_alpha.min(), log_alpha.max())
