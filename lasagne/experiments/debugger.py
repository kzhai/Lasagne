import logging
import numpy
import os
import theano
import theano.tensor as T
import timeit

from lasagne import layers
from lasagne import nonlinearities, objectives, updates, Xregularization
from ..layers import DenseLayer, BernoulliDropoutLayer, BernoulliDropoutLayer

logger = logging.getLogger(__name__)

__all__ = [
	"subsample_dataset",
	"display_architecture",
	"debug_function_output",
	#
	"snapshot_dropouts",
	"snapshot_conv_filters"
]


def _subsample_dataset(dataset, fraction=0.01):
	if dataset is None:
		return None
	dataset_x, dataset_y = dataset
	size = len(dataset_y)
	# indices = numpy.random.permutation(size)[:int(size * fraction)]
	indices = numpy.arange(size)[:int(size * fraction)]
	dataset_x = dataset_x[indices]
	dataset_y = dataset_y[indices]
	return dataset_x, dataset_y


def subsample_dataset(train_dataset, validate_dataset, test_dataset, fraction=0.01, **kwargs):
	if validate_dataset is None:
		size_before = [len(train_dataset[1]), 0, len(test_dataset[1])]
	else:
		size_before = [len(train_dataset[1]), len(validate_dataset[1]), len(test_dataset[1])]
	train_dataset = _subsample_dataset(train_dataset, fraction)
	validate_dataset = _subsample_dataset(validate_dataset, fraction)
	test_dataset = _subsample_dataset(test_dataset, fraction)
	if validate_dataset is None:
		size_after = [len(train_dataset[1]), 0, len(test_dataset[1])]
	else:
		size_after = [len(train_dataset[1]), len(validate_dataset[1]), len(test_dataset[1])]
	logger.debug("debug: subsample [train, validate, test] sets from %s to %s instances" % (size_before, size_after))
	print("debug: subsample [train, validate, test] sets from %s to %s instances" % (size_before, size_after))
	return train_dataset, validate_dataset, test_dataset


def display_architecture(network, **kwargs):
	input_shape = network._input_shape
	for layer in network.get_network_layers():
		logger.debug("debug: output size after %s is %s" % (layer, layers.get_output_shape(layer, input_shape)))
		print("debug: output size after %s is %s" % (layer, layers.get_output_shape(layer, input_shape)))


def debug_function_output(network, minibatch, **kwargs):
	# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
	nondeterministic_loss = network.get_loss(network._output_variable)
	nondeterministic_objective = network.get_objectives(network._output_variable)
	nondeterministic_accuracy = network.get_objectives(network._output_variable,
													   objective_functions="categorical_accuracy")

	# Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
	deterministic_loss = network.get_loss(network._output_variable, deterministic=True)
	deterministic_objective = network.get_objectives(network._output_variable, deterministic=True)
	deterministic_accuracy = network.get_objectives(network._output_variable,
													objective_functions="categorical_accuracy", deterministic=True)

	function_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=[nondeterministic_loss, nondeterministic_objective, nondeterministic_accuracy,
				 deterministic_loss, deterministic_objective, deterministic_accuracy],
		on_unused_input='raise'
	)

	minibatch_x, minibatch_y = minibatch
	debugger_function_output = function_debugger(minibatch_x, minibatch_y)

	logger.debug("stochastic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
				 (debugger_function_output[0], debugger_function_output[1],
				  debugger_function_output[0] - debugger_function_output[1], debugger_function_output[2] * 100))
	print("debug: stochastic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
		  (debugger_function_output[0], debugger_function_output[1],
		   debugger_function_output[0] - debugger_function_output[1], debugger_function_output[2] * 100))

	logger.debug("deterministic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
				 (debugger_function_output[3], debugger_function_output[4],
				  debugger_function_output[3] - debugger_function_output[4], debugger_function_output[5] * 100))
	print("debug: deterministic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
		  (debugger_function_output[3], debugger_function_output[4],
		   debugger_function_output[3] - debugger_function_output[4], debugger_function_output[5] * 100))


def debug_rademacher_p_2_q_2(network, minibatch, rescale=False, **kwargs):
	input_layer = Xregularization.find_input_layer(network)
	input_shape = layers.get_output_shape(input_layer)
	input_value = layers.get_output(input_layer)
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])
	dummy, k = network.get_output_shape()

	output, mapping = [], []
	mapping.append("k * sqrt(log(d) / n)")
	output.append(k * T.sqrt(T.log(d) / n))
	mapping.append("max(abs(input_value))")
	output.append(T.max(abs(input_value)))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, BernoulliDropoutLayer):
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			mapping.append("sqrt(sum(retain_probability ** 2))")
			output.append(T.sqrt(T.sum(retain_probability ** 2)))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			mapping.append("max(sqrt(sum(layer.W ** 2, axis=0)))")
			output.append(T.max(T.sqrt(T.sum(layer.W ** 2, axis=0))))

			if rescale:
				d1, d2 = layer.W.shape
				mapping.append("1. / (d1 * sqrt(log(d2)))")
				output.append(1. / (d1 * T.sqrt(T.log(d2))))
				# this is to offset glorot initialization
				mapping.append("sqrt(d1 + d2)")
				output.append(T.sqrt(d1 + d2))

	function_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=output,
		on_unused_input='warn'
	)

	minibatch_x, minibatch_y = minibatch
	debugger_function_outputs = function_debugger(minibatch_x, minibatch_y)
	'''
	for token_mapping, token_output in zip(mapping, debugger_function_outputs):
		print("debug: %s = %g" % (token_mapping, token_output))
	'''
	logger.debug("Rademacher (p=2, q=2) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))
	print("debug: Rademacher (p=2, q=2) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))


debug_rademacher = debug_rademacher_p_2_q_2


def debug_rademacher_p_1_q_inf(network, minibatch, rescale=False, **kwargs):
	input_layer = Xregularization.find_input_layer(network)
	input_shape = layers.get_output_shape(input_layer)
	input_value = layers.get_output(input_layer)
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])
	dummy, k = network.get_output_shape()

	output, mapping = [], []
	mapping.append("k * sqrt(log(d) / n)")
	output.append(k * T.sqrt(T.log(d) / n))
	mapping.append("max(abs(input_value))")
	output.append(T.max(abs(input_value)))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, BernoulliDropoutLayer):
			# retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1)
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			mapping.append("max(abs(retain_probability))")
			output.append(T.max(abs(retain_probability)))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			mapping.append("max(sum(abs(layer.W), axis=0))")
			output.append(T.max(T.sum(abs(layer.W), axis=0)))

			if rescale:
				d1, d2 = layer.W.shape
				mapping.append("1. / (d1 * sqrt(log(d2)))")
				output.append(1. / (d1 * T.sqrt(T.log(d2))))
				mapping.append("sqrt(d1 + d2)")
				output.append(T.sqrt(d1 + d2))

	function_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=output,
		on_unused_input='warn'
	)

	minibatch_x, minibatch_y = minibatch
	debugger_function_outputs = function_debugger(minibatch_x, minibatch_y)
	'''
	for token_mapping, token_output in zip(mapping, debugger_function_outputs):
		print("debug: %s = %g" % (token_mapping, token_output))
	'''
	logger.debug("Rademacher (p=1, q=inf) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))
	print("debug: Rademacher (p=1, q=inf) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))


def debug_rademacher_p_inf_q_1(network, minibatch, rescale=False, **kwargs):
	input_layer = Xregularization.find_input_layer(network)
	input_shape = layers.get_output_shape(input_layer)
	input_value = layers.get_output(input_layer)
	n = network._input_variable.shape[0]
	d = T.prod(input_shape[1:])
	dummy, k = network.get_output_shape()

	output, mapping = [], []
	mapping.append("k * sqrt(log(d) / n)")
	output.append(k * T.sqrt(T.log(d) / n))
	mapping.append("max(abs(input_value))")
	output.append(T.max(abs(input_value)))

	for layer in network.get_network_layers():
		if isinstance(layer, BernoulliDropoutLayer) or isinstance(layer, BernoulliDropoutLayer):
			# retain_probability = numpy.clip(layer.activation_probability.eval(), 0, 1)
			retain_probability = T.clip(layer.activation_probability, 0, 1)
			mapping.append("sum(abs(retain_probability))")
			output.append(T.sum(abs(retain_probability)))
		elif isinstance(layer, DenseLayer):
			# compute B_l * p_l, with a layer-wise scale constant
			mapping.append("max(abs(layer.W))")
			output.append(T.max(abs(layer.W)))

			if rescale:
				d1, d2 = layer.W.shape
				mapping.append("1. / d1")
				output.append(1. / d1)
				mapping.append("sqrt(d1 + d2)")
				output.append(T.sqrt(d1 + d2))

	function_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=output,
		on_unused_input='warn'
	)

	minibatch_x, minibatch_y = minibatch
	debugger_function_outputs = function_debugger(minibatch_x, minibatch_y)
	'''
	for token_mapping, token_output in zip(mapping, debugger_function_outputs):
		print("debug: %s = %g" % (token_mapping, token_output))
	'''
	logger.debug("Rademacher (p=inf, q=1) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))
	print("debug: Rademacher (p=inf, q=1) complexity: regularizer=%g" % (numpy.prod(debugger_function_outputs)))


def snapshot_dropouts(network, settings=None, **kwargs):
	dropout_layer_index = 0
	for network_layer in network.get_network_layers():
		if isinstance(network_layer, layers.BernoulliDropoutLayer) or \
				isinstance(network_layer, layers.BernoulliDropoutLayer) or \
				isinstance(network_layer, layers.DynamicDropoutLayer):
			layer_retain_probability = network_layer.activation_probability.eval()
		elif isinstance(network_layer, layers.SparseVariationalDropoutLayer):
			alpha = T.exp(network_layer.log_alpha).eval()
			layer_retain_probability = 1. / (1. + alpha)
		elif isinstance(network_layer, layers.VariationalDropoutTypeALayer) or \
				isinstance(network_layer, layers.VariationalDropoutTypeBLayer):
			sigma = T.nnet.sigmoid(network_layer.logit_sigma).eval()
			layer_retain_probability = 1. / (1. + sigma ** 2)
		elif isinstance(network_layer, layers.GaussianDropoutLayer) or \
				isinstance(network_layer, layers.FastDropoutLayer):
			sigma = T.nnet.sigmoid(network_layer.logit_sigma).eval()
			layer_retain_probability = 1. / (1. + sigma ** 2)
		else:
			continue

		print("retain rates: epoch %i, shape %s, average %f, minimum %f, maximum %f" % (
			network.epoch_index, layer_retain_probability.shape,
			numpy.mean(layer_retain_probability),
			numpy.min(layer_retain_probability),
			numpy.max(layer_retain_probability)))

		if settings is not None:
			# layer_retain_probability = numpy.reshape(layer_retain_probability, numpy.prod(layer_retain_probability.shape))
			retain_rate_file = os.path.join(settings.output_directory,
											"noise.%d.epoch.%d.npy" % (dropout_layer_index, network.epoch_index))
			numpy.save(retain_rate_file, layer_retain_probability)
		dropout_layer_index += 1


def snapshot_conv_filters(network, settings, **kwargs):
	conv_layer_index = 0
	for network_layer in network.get_network_layers():
		if isinstance(network_layer, layers.Conv2DLayer):
			conv_filters = network_layer.W.eval()
		else:
			continue

		# conv_filters = numpy.reshape(conv_filters, numpy.prod(conv_filters.shape))
		conv_filter_file = os.path.join(settings.output_directory,
										"conv.%d.epoch.%d.npy" % (conv_layer_index, network.epoch_index))
		numpy.save(conv_filter_file, conv_filters)

		conv_layer_index += 1


def main():
	from ..experiments import load_mnist

	train_set_x, train_set_y, validate_set_x, validate_set_y, test_set_x, test_set_y = load_mnist();
	train_set_x = numpy.reshape(train_set_x, (train_set_x.shape[0], numpy.prod(train_set_x.shape[1:])))
	validate_set_x = numpy.reshape(validate_set_x, (validate_set_x.shape[0], numpy.prod(validate_set_x.shape[1:])))
	test_set_x = numpy.reshape(test_set_x, (test_set_x.shape[0], numpy.prod(test_set_x.shape[1:])))

	train_dataset = train_set_x, train_set_y
	validate_dataset = validate_set_x, validate_set_y
	test_dataset = test_set_x, test_set_y

	input_shape = list(train_set_x.shape[1:]);
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)

	'''
	network=MultiLayerPerceptron(
		incoming = input_shape,

		layer_dimensions = [32, 64, 10],
		layer_nonlinearities = [nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_parameters = [0.8, 0.5, 0.5],
		layer_activation_styles = ["bernoulli", "bernoulli", "bernoulli"],

		objective_functions = objectives.categorical_crossentropy,
		update_function = updates.nesterov_momentum,

		learning_rate = 0.001,
		learning_rate_decay_style = None,
		learning_rate_decay_parameter = 0,
		validation_interval = 1000,
	)
	'''

	network = DynamicMultiLayerPerceptron(
		incoming=input_shape,

		layer_dimensions=[32, 64, 10],
		layer_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_parameters=[0.8, 0.5, 0.5],
		layer_activation_styles=["bernoulli", "bernoulli", "bernoulli"],

		objective_functions=objectives.categorical_crossentropy,
		update_function=updates.nesterov_momentum,

		learning_rate=0.001,
		learning_rate_decay_style=None,
		learning_rate_decay_parameter=0,
		dropout_rate_update_interval=10,
		validation_interval=1000,
	)

	regularizer_functions = {};
	# regularizer_functions[regularization.l1] = 0.1
	# regularizer_functions[regularization.l2] = [0.2, 0.5, 0.4]
	# regularizer_functions[regularization.rademacher] = 1e-6
	network.set_regularizers(regularizer_functions)

	########################
	# START MODEL TRAINING #
	########################

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	number_of_epochs = 10;
	minibatch_size = 1000;
	for epoch_index in range(number_of_epochs):
		network.train(train_dataset, minibatch_size, validate_dataset, test_dataset);
		print "PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs);
	end_train = timeit.default_timer()

	print "Optimization complete..."
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
	print 'The code finishes in %.2fm' % ((end_train - start_train) / 60.)


if __name__ == '__main__':
	main();
