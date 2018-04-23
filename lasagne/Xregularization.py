import numpy
import theano.tensor as T

from .layers import get_output, get_output_shape, Layer, EmbeddingLayer, DenseLayer, ObstructedDenseLayer, \
	AdaptiveDropoutLayer


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
		if isinstance(layer_2, AdaptiveDropoutLayer) or isinstance(layer_2, AdaptiveDropoutLayer):
			return layer_1


def __find_input_layer(network):
	for layer in network.get_network_layers():
		if isinstance(layer, EmbeddingLayer):
			return layer
	return network._input_layer


find_input_layer = __find_pre_dropout_layer


# find_input_layer = __find_input_layer

def rademacher_p_2_q_2(network, **kwargs):
	kwargs['rescale'] = kwargs.get('rescale', True)

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
		if isinstance(layer, AdaptiveDropoutLayer):
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			# rademacher_regularization *= T.sqrt(T.sum(retain_probability ** 2))
			rademacher_regularization *= T.sqrt(T.sum(abs(retain_probability)))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			rademacher_regularization *= 2 * T.max(T.sqrt(T.sum(layer.W ** 2, axis=0)))

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


def rademacher_p_inf_q_1(network, **kwargs):
	kwargs['rescale'] = kwargs.get('rescale', True)

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
		if isinstance(layer, AdaptiveDropoutLayer):
			# retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1)
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			rademacher_regularization *= T.sum(abs(retain_probability))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			rademacher_regularization *= 2 * T.max(abs(layer.W))

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


rademacher = rademacher_p_inf_q_1


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

	k1 = 0.63576
	k2 = 1.8732
	k3 = 1.48695
	C = -k1

	params = network.get_network_params()
	log_alphas = [T.nnet.sigmoid(p) for p in params if p.name == "variational.dropout.logit_sigma"]

	return -T.sum(
		[T.sum(k1 * T.nnet.sigmoid(k2 + (k3 * log_alpha)) - 0.5 * T.log(1 + T.pow(T.exp(log_alpha), -1)) + C) for
		 log_alpha in log_alphas])


'''
def mclog_likelihood(N=None,
                     base_likelihood=lasagne.objectives.categorical_crossentropy):
    return lambda predictions, targets: N * base_likelihood(predictions, targets)
'''


class GaussianMixturePrior(Layer):
	"""A Gaussian Mixture prior for Neural Networks """

	def __init__(self, network, number_of_components, pretrained_weights, pi_zero=1 - 1e-3, **kwargs):
		self.neural_network = network
		self.number_of_components = number_of_components
		self.network_weights = [K.flatten(w) for w in network_weights]
		# self.pretrained_weights = special_flatten(pretrained_weights)
		self.pi_zero = pi_zero

		# super(GaussianMixturePrior, self).__init__(**kwargs)

		number_of_components = self.number_of_components

		# create trainable ...
		#    ... means
		init_mean = numpy.linspace(-0.6, 0.6, self.number_of_components - 1)
		self.means = self.add_param(init_mean, init_mean.shape, name="means", trainable=True, regularizable=False)
		# self.means = K.variable(init_mean, name='means')
		#   ... the variance (we will work in log-space for more stability)
		init_stds = numpy.tile(0.25, self.number_of_components)
		init_gamma = - numpy.log(numpy.power(init_stds, 2))
		self.gammas = self.add_param(init_gamma, init_gamma.shape, name="gammas", trainable=True, regularizable=False)
		#   ... the mixing proportions
		init_mixing_proportions = numpy.ones((self.number_of_components - 1))
		init_mixing_proportions *= (1. - self.pi_zero) / (self.number_of_components - 1)
		self.rhos = self.add_param(numpy.log(init_mixing_proportions), init_mixing_proportions.shape, name="rhos",
		                           trainable=True, regularizable=False)

		print("means", self.means, init_mean)
		print("gammas", self.gammas, init_gamma)
		print("rhos", self.rhos, init_mixing_proportions)

		# Finally, add the variables to the trainable parameters
		self.trainable_weights = [self.means] + [self.gammas] + [self.rhos]

	def call(self, x, mask=None):
		print("---------->", "checkpoint inside call")
		number_of_components = self.number_of_components
		loss = 0
		# here we stack together the trainable and non-trainable params
		#     ... the mean vector
		means = T.concatenate([T.scalar(0.), self.means], axis=0)
		#     ... the variances
		precision = T.exp(self.gammas)
		#     ... the mixing proportions (we are using the log-sum-exp trick here)
		min_rho = T.min(self.rhos)
		mixing_proportions = T.exp(self.rhos - min_rho)
		mixing_proportions = (1 - self.pi_zero) * mixing_proportions / T.sum(mixing_proportions)
		mixing_proportions = T.concatenate([T.scalar(self.pi_zero), mixing_proportions], axis=0)

		# compute the loss given by the gaussian mixture
		for weights in self.network_weights:
			loss = loss + self.compute_loss(weights, mixing_proportions, means, precision)

		# GAMMA PRIOR ON PRECISION
		# ... for the zero component
		(alpha, beta) = (5e3, 20e-1)
		neglogprop = (1 - alpha) * self.gammas[0] + beta * precision[0]
		loss = loss + T.sum(neglogprop)
		# ... and all other components
		alpha, beta = (2.5e2, 1e-1)
		idx = numpy.arange(1, number_of_components)
		neglogprop = (1 - alpha) * self.gammas[idx] + beta * precision[idx]
		loss = loss + T.sum(neglogprop)

		return loss

	def compute_loss(self, weights, mixing_proportions, means, precision):
		diff = weights[:, None] - means  # shape: (nb_params, nb_components)
		unnormalized_log_likelihood = - (diff ** 2) / 2 * T.flatten(precision)
		Z = T.sqrt(precision / (2 * numpy.pi))
		log_likelihood = logsumexp(unnormalized_log_likelihood, w=T.flatten(mixing_proportions * Z), axis=1)

		# return the neg. log-likelihood for the prior
		return - T.sum(log_likelihood)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], 1)


def logsumexp(t, w=None, axis=1):
	"""
	t... tensor
	w... weight tensor
	"""

	t_max = T.max(t, axis=axis, keepdims=True)

	if w is not None:
		tmp = w * T.exp(t - t_max)
	else:
		tmp = T.exp(t - t_max)

	out = T.sum(tmp, axis=axis)
	out = T.log(out)

	t_max = T.max(t, axis=axis)

	return out + t_max
