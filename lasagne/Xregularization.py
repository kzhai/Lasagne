import theano.tensor as T

from .layers import get_output, get_output_shape, EmbeddingLayer, DenseLayer, ObstructedDenseLayer, \
	BernoulliDropoutLayer, AdaptiveDropoutLayer


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
	return sum(coeff * sum(penalty(layer_tuple, tags, **kwargs)) for layer_tuple, coeff in list(layers.items()))


def l1_norm(X, axis=None):
	return T.sum(abs(X), axis=axis)


def l2_norm(X, axis=None):
	return T.sqrt(T.sum(X ** 2, axis=axis))


def linf_norm(X, axis=None):
	return T.max(abs(X), axis=axis)


def __find_pre_dropout_layer(network):
	for layer_1, layer_2 in zip(network.get_network_layers()[:-1], network.get_network_layers()[1:]):
		if isinstance(layer_2, BernoulliDropoutLayer) or isinstance(layer_2, AdaptiveDropoutLayer):
			return layer_1


def __find_input_layer(network):
	for layer in network.get_network_layers():
		if isinstance(layer, EmbeddingLayer):
			return layer
	return network._input_layer


find_input_layer = __find_pre_dropout_layer


# find_input_layer = __find_input_layer

def rademacher_p_2_q_2(network, **kwargs):
	kwargs['rescale'] = kwargs.get('rescale', False)

	input_layer = find_input_layer(network)

	input_shape = get_output_shape(input_layer)
	input_value = get_output(input_layer)
	# n = input_shape[0]
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])

	# pseudo_input_layer = __find_layer_before_dropout(network)
	# print pseudo_input_layer

	# d = T.prod(network._input_variable.shape[1:])
	# d = T.prod(get_output_shape(pseudo_input_layer)[1:])
	# n, d = network._input_variable.shape
	dummy, k = network.get_output_shape()
	rademacher_regularization = k * T.sqrt(T.log(d) / n)
	rademacher_regularization *= T.max(abs(input_value))
	# rademacher_regularization *= T.max(abs(network._input_variable))
	# rademacher_regularization *= T.max(abs(get_output(pseudo_input_layer)))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			rademacher_regularization *= T.sqrt(T.sum(retain_probability ** 2))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			rademacher_regularization *= T.max(T.sqrt(T.sum(layer.W ** 2, axis=0)))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				rademacher_regularization /= d1 * T.sqrt(T.log(d2))
				rademacher_regularization *= T.sqrt(d1 + d2)
		elif isinstance(layer, ObstructedDenseLayer):
			rademacher_regularization *= T.max(T.sqrt(T.sum(layer.W ** 2, axis=0)))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				d2 = layer.num_units
				rademacher_regularization /= d1 * T.sqrt(T.log(d2))
				rademacher_regularization *= T.sqrt(d1 + d2)

	return rademacher_regularization


rademacher = rademacher_p_2_q_2  # shortcut


def rademacher_p_inf_q_1(network, **kwargs):
	kwargs['rescale'] = kwargs.get('rescale', False)

	input_layer = find_input_layer(network)

	input_shape = get_output_shape(input_layer)
	input_value = get_output(input_layer)
	# n = input_shape[0]
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])

	# d = T.prod(network._input_variable.shape[1:])
	# n, d = network._input_variable.shape
	dummy, k = network.get_output_shape()
	rademacher_regularization = k * T.sqrt(T.log(d) / n)
	# rademacher_regularization *= T.max(abs(network._input_variable))
	rademacher_regularization *= T.max(abs(input_value))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
			# retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1)
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			rademacher_regularization *= T.sum(abs(retain_probability))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			rademacher_regularization *= T.max(abs(layer.W))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				rademacher_regularization /= d1
				rademacher_regularization *= T.sqrt(d1 + d2)
		elif isinstance(layer, ObstructedDenseLayer):
			rademacher_regularization *= T.max(abs(layer.W))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				d2 = layer.num_units
				rademacher_regularization /= d1
				rademacher_regularization *= T.sqrt(d1 + d2)

	return rademacher_regularization


def rademacher_p_1_q_inf(network, **kwargs):
	kwargs['rescale'] = kwargs.get('rescale', False)

	input_layer = find_input_layer(network)

	input_shape = get_output_shape(input_layer)
	input_value = get_output(input_layer)
	# n = input_shape[0]
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])

	# d = T.prod(network._input_variable.shape[1:])
	# n, d = network._input_variable.shape
	dummy, k = network.get_output_shape()
	rademacher_regularization = k * T.sqrt(T.log(d) / n)
	rademacher_regularization *= T.max(abs(input_value))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, AdaptiveDropoutLayer):
			# retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1)
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			rademacher_regularization *= T.max(abs(retain_probability))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			rademacher_regularization *= T.max(T.sum(abs(layer.W), axis=0))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				rademacher_regularization /= d1 * T.sqrt(T.log(d2))
				rademacher_regularization *= T.sqrt(d1 + d2)
		elif isinstance(layer, ObstructedDenseLayer):
			rademacher_regularization *= T.max(T.sum(abs(layer.W), axis=0))

			if kwargs['rescale']:
				# this is to offset glorot initialization
				d1, d2 = layer.W.shape
				d2 = layer.num_units
				rademacher_regularization /= d1 * T.sqrt(T.log(d2))
				rademacher_regularization *= T.sqrt(d1 + d2)

	return rademacher_regularization


# first part is the KL divergence of the priors
def kl_divergence_kingma(network, **kwargs):
	"""
	Has the same interface as L2 regularisation in lasagne.
	Input:
		* output_layer - final layer in your network, used to pull the
		weights out of every other layer
	Output:
		* Theano expression for the KL divergence on the priors:
		- D_{KL}( q_{\phi}(w) || p(w) )
	"""
	# gather up all the alphas

	# params = get_all_params(network)
	# params = network.get_network_params(name="variational.dropout.logit_sigma")
	# alphas = [T.nnet.sigmoid(p) for p in params]
	params = network.get_network_params()
	alphas = [T.nnet.sigmoid(p) for p in params if p.name == "variational.dropout.logit_sigma"]

	# I hope all these decimal places are important
	c1 = 1.161451241083230
	c2 = -1.502041176441722
	c3 = 0.586299206427007

	# will get taken apart again in the autodiff
	return -T.sum([0.5 * T.sum(T.log(alpha)) + c1 * T.sum(alpha) + c2 * T.sum(T.pow(alpha, 2))
	               + c3 * T.sum(T.pow(alpha, 3)) for alpha in alphas])


def kl_divergence_sparse(network, **kwargs):
	"""
	Based on the paper "Variational Dropout Sparsifies Deep Neural Networks" by
	Dmitry Molchanov, Arsenii Ashukha and Dmitry Vetrov, https://arxiv.org/abs/1701.05369.
	Modification so that we don't need to constrain alpha to be below 1. Then,
	the network is free to drop units that are not useful.
	Input:
		* output_layer - final layer in a Lasagne network, so we can pull  the
		alpha values out of all the layers
	"""
	# gather up all the alphas
	# params = network.get_network_params(name="variational.dropout.log_alpha")
	# alphas = [T.exp(p) for p in params]
	params = network.get_network_params()
	alphas = [T.exp(p) for p in params if p.name == "variational.dropout.log_alpha"]

	k1 = 0.63576
	k2 = 1.8732
	k3 = 1.48695
	C = -k1

	return -T.sum(
		[T.sum(k1 * T.nnet.sigmoid(k2 + (k3 * T.log(alpha))) - 0.5 * T.log(1 + T.pow(alpha, -1)) + C) for alpha in
		 alphas])


'''
def mclog_likelihood(N=None,
                     base_likelihood=lasagne.objectives.categorical_crossentropy):
    return lambda predictions, targets: N * base_likelihood(predictions, targets)
'''
