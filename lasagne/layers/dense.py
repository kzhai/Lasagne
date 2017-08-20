import numpy as np
import theano.tensor as T

from .base import Layer
from .. import init
from .. import nonlinearities

__all__ = [
	"DenseLayer",
	"NINLayer",
	#
	#
	#
	#
	#
	"ElasticDenseLayer",
]


class DenseLayer(Layer):
	"""
	lasagne.layers.DenseLayer(incoming, num_units,
	W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
	nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=1, **kwargs)

	A fully connected layer.

	Parameters
	----------
	incoming : a :class:`Layer` instance or a tuple
		The layer feeding into this layer, or the expected input shape

	num_units : int
		The number of units of the layer

	W : Theano shared variable, expression, numpy array or callable
		Initial value, expression or initializer for the weights.
		These should be a matrix with shape ``(num_inputs, num_units)``.
		See :func:`lasagne.utils.create_param` for more information.

	b : Theano shared variable, expression, numpy array, callable or ``None``
		Initial value, expression or initializer for the biases. If set to
		``None``, the layer will have no biases. Otherwise, biases should be
		a 1D array with shape ``(num_units,)``.
		See :func:`lasagne.utils.create_param` for more information.

	nonlinearity : callable or None
		The nonlinearity that is applied to the layer activations. If None
		is provided, the layer will be linear.

	num_leading_axes : int
		Number of leading axes to distribute the dot product over. These axes
		will be kept in the output tensor, remaining axes will be collapsed and
		multiplied against the weight matrix. A negative number gives the
		(negated) number of trailing axes to involve in the dot product.

	Examples
	--------
	>>> from lasagne.layers import InputLayer, DenseLayer
	>>> l_in = InputLayer((100, 20))
	>>> l1 = DenseLayer(l_in, num_units=50)

	If the input has more than two axes, by default, all trailing axes will be
	flattened. This is useful when a dense layer follows a convolutional layer.

	>>> l_in = InputLayer((None, 10, 20, 30))
	>>> DenseLayer(l_in, num_units=50).output_shape
	(None, 50)

	Using the `num_leading_axes` argument, you can specify to keep more than
	just the first axis. E.g., to apply the same dot product to each step of a
	batch of time sequences, you would want to keep the first two axes.

	>>> DenseLayer(l_in, num_units=50, num_leading_axes=2).output_shape
	(None, 10, 50)
	>>> DenseLayer(l_in, num_units=50, num_leading_axes=-1).output_shape
	(None, 10, 20, 50)
	"""

	def __init__(self, incoming, num_units, W=init.GlorotUniform(),
	             b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
	             num_leading_axes=1, **kwargs):
		super(DenseLayer, self).__init__(incoming, **kwargs)
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
		num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

		self.W = self.add_param(W, (num_inputs, num_units), name="W")
		if b is None:
			self.b = None
		else:
			self.b = self.add_param(b, (num_units,), name="b",
			                        regularizable=False)

	def get_output_shape_for(self, input_shape):
		return input_shape[:self.num_leading_axes] + (self.num_units,)

	def get_output_for(self, input, **kwargs):
		num_leading_axes = self.num_leading_axes
		if num_leading_axes < 0:
			num_leading_axes += input.ndim
		if input.ndim > num_leading_axes + 1:
			# flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
			input = input.flatten(num_leading_axes + 1)

		activation = T.dot(input, self.W)
		if self.b is not None:
			activation = activation + self.b
		return self.nonlinearity(activation)


class NINLayer(Layer):
	"""
	lasagne.layers.NINLayer(incoming, num_units, untie_biases=False,
	W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
	nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

	Network-in-network layer.
	Like DenseLayer, but broadcasting across all trailing dimensions beyond the
	2nd.  This results in a convolution operation with filter size 1 on all
	trailing dimensions.  Any number of trailing dimensions is supported,
	so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.

	Parameters
	----------
	incoming : a :class:`Layer` instance or a tuple
		The layer feeding into this layer, or the expected input shape

	num_units : int
		The number of units of the layer

	untie_biases : bool
		If false the network has a single bias vector similar to a dense
		layer. If true a separate bias vector is used for each trailing
		dimension beyond the 2nd.

	W : Theano shared variable, expression, numpy array or callable
		Initial value, expression or initializer for the weights.
		These should be a matrix with shape ``(num_inputs, num_units)``,
		where ``num_inputs`` is the size of the second dimension of the input.
		See :func:`lasagne.utils.create_param` for more information.

	b : Theano shared variable, expression, numpy array, callable or ``None``
		Initial value, expression or initializer for the biases. If set to
		``None``, the layer will have no biases. Otherwise, biases should be
		a 1D array with shape ``(num_units,)`` for ``untie_biases=False``, and
		a tensor of shape ``(num_units, input_shape[2], ..., input_shape[-1])``
		for ``untie_biases=True``.
		See :func:`lasagne.utils.create_param` for more information.

	nonlinearity : callable or None
		The nonlinearity that is applied to the layer activations. If None
		is provided, the layer will be linear.

	Examples
	--------
	>>> from lasagne.layers import InputLayer, NINLayer
	>>> l_in = InputLayer((100, 20, 10, 3))
	>>> l1 = NINLayer(l_in, num_units=5)

	References
	----------
	.. [1] Lin, Min, Qiang Chen, and Shuicheng Yan (2013):
		   Network in network. arXiv preprint arXiv:1312.4400.
	"""

	def __init__(self, incoming, num_units, untie_biases=False,
	             W=init.GlorotUniform(), b=init.Constant(0.),
	             nonlinearity=nonlinearities.rectify, **kwargs):
		super(NINLayer, self).__init__(incoming, **kwargs)
		self.nonlinearity = (nonlinearities.identity if nonlinearity is None
		                     else nonlinearity)

		self.num_units = num_units
		self.untie_biases = untie_biases

		num_input_channels = self.input_shape[1]

		self.W = self.add_param(W, (num_input_channels, num_units), name="W")
		if b is None:
			self.b = None
		else:
			if self.untie_biases:
				biases_shape = (num_units,) + self.output_shape[2:]
			else:
				biases_shape = (num_units,)
			self.b = self.add_param(b, biases_shape, name="b",
			                        regularizable=False)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.num_units) + input_shape[2:]

	def get_output_for(self, input, **kwargs):
		# cf * bc01... = fb01...
		out_r = T.tensordot(self.W, input, axes=[[0], [1]])
		# input dims to broadcast over
		remaining_dims = range(2, input.ndim)
		# bf01...
		out = out_r.dimshuffle(1, 0, *remaining_dims)

		if self.b is None:
			activation = out
		else:
			if self.untie_biases:
				# no broadcast
				remaining_dims_biases = range(1, input.ndim - 1)
			else:
				remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
			b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
			activation = out + b_shuffled

		return self.nonlinearity(activation)


#
#
#
#
#

'''
class DenseLayer(Layer):
    def __init__(self, incoming, num_units, Wfc, nonlinearity=rectify, mnc=False, b=Constant(0.), **kwargs):
        super(DenseLayer, self).__init__(incoming)
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.W = self.add_param(Wfc, (self.num_inputs, self.num_units), name="W")
        if mnc:
            self.W = updates.norm_constraint(self.W, mnc)

        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        return self.get_output_for_(input, deterministic, **kwargs)

    def get_output_for_(self, input, deterministic, **kwargs):
        return self.nonlinearity(T.dot(input, self.W) + self.b)
'''


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
		if not (deterministic or np.all(retain_prob == 1)):
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
		log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
		return '%.4f' % (np.sum(log_alpha > thresh) * 1.0 / log_alpha.size)

	def get_reg(self):
		log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
		return '%.1f, %.1f' % (log_alpha.min(), log_alpha.max())


class ElasticDenseLayer(DenseLayer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(),
	             b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
	             num_leading_axes=1, **kwargs):
		super(ElasticDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, num_leading_axes, **kwargs)

	# self.set_output(num_units, W=init.GlorotUniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify)

	def set_input(self, incoming, W=init.GlorotUniform(), num_leading_axes=1):
		if isinstance(incoming, tuple):
			self.input_shape = incoming
			self.input_layer = None
		else:
			self.input_shape = incoming.output_shape
			self.input_layer = incoming

		'''
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

		if any(d is not None and d <= 0 for d in self.input_shape):
			raise ValueError((
				                 "Cannot create Layer with a non-positive input_shape "
				                 "dimension. input_shape=%r, self.name=%r") % (
				                 self.input_shape, self.name))
		'''

		self.set_W(W)

	def set_output(self, num_units, W=init.GlorotUniform(), b=init.Constant(0.)):
		self.num_units = num_units

		self.set_W(W)
		self.set_b(b)

	def set_W(self, W=init.GlorotUniform()):
		old_W = self.W.eval()
		self.params.pop(self.W);

		num_inputs = int(np.prod(self.input_shape[self.num_leading_axes:]))
		self.W = self.add_param(W, (num_inputs, self.num_units), name="W")
		return old_W

	def set_b(self, b=init.Constant(0.)):
		old_b = self.b.eval();
		self.params.pop(self.b);

		if b is None:
			self.b = None
		else:
			self.b = self.add_param(b, (self.num_units,), name="b", regularizable=False)
		return old_b

	'''
	def del_param(self, spec, shape, name=None, **tags):
		# prefix the param name with the layer name if it exists
		if name is not None:
			if self.name is not None:
				name = "%s.%s" % (self.name, name)
		# create shared variable, or pass through given variable/expression
		param = utils.create_param(spec, shape, name)
		# parameters should be trainable and regularizable by default
		tags['trainable'] = tags.get('trainable', True)
		tags['regularizable'] = tags.get('regularizable', True)
		self.params[param] = set(tag for tag, value in tags.items() if value)

		return param
	'''
