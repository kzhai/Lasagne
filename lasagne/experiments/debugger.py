import logging
import os
import timeit

import numpy
import theano
import theano.tensor as T

from lasagne import layers
from lasagne import nonlinearities, objectives, updates, Xregularization
from ..layers import DenseLayer, AdaptiveDropoutLayer

logger = logging.getLogger(__name__)

__all__ = [
	"subsample_dataset",
	"display_architecture",
	"debug_function_output",
	#
	"snapshot_dropout",
	"snapshot_dense",
	"snapshot_convolution"
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


def debug_function_output(network, dataset, **kwargs):
	dataset_x, dataset_y = dataset

	number_of_data = dataset_x.shape[0]
	data_indices = numpy.random.permutation(number_of_data)

	minibatch_size = kwargs.get('minibatch_size', number_of_data)
	output_file = kwargs.get('output_file', None)

	# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
	stochastic_loss = network.get_loss(network._output_variable)
	stochastic_objective = network.get_objectives(network._output_variable)
	stochastic_accuracy = network.get_objectives(network._output_variable,
	                                             objective_functions="categorical_accuracy")

	# Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
	deterministic_loss = network.get_loss(network._output_variable, deterministic=True)
	deterministic_objective = network.get_objectives(network._output_variable, deterministic=True)
	deterministic_accuracy = network.get_objectives(network._output_variable,
	                                                objective_functions="categorical_accuracy", deterministic=True)

	stochastic_function_output_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=[stochastic_accuracy, stochastic_loss, stochastic_objective],
		on_unused_input='raise'
	)

	deterministic_function_output_debugger = theano.function(
		inputs=[network._input_variable, network._output_variable],
		outputs=[deterministic_accuracy, deterministic_loss, deterministic_objective],
		on_unused_input='raise'
	)

	minibatch_start_index = 0
	minibatch_index = 0
	minibatch_outputs = numpy.zeros((int(numpy.ceil(1. * number_of_data / minibatch_size)), 9))

	while minibatch_start_index < number_of_data:
		# automatically handles the left-over data
		minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

		minibatch_x = dataset_x[minibatch_indices, :]
		minibatch_y = dataset_y[minibatch_indices]

		function_stochastic_output = stochastic_function_output_debugger(minibatch_x, minibatch_y)
		function_deterministic_output = deterministic_function_output_debugger(minibatch_x, minibatch_y)

		function_stochastic_output.append(function_stochastic_output[-2] - function_stochastic_output[-1])
		function_deterministic_output.append(function_deterministic_output[-2] - function_deterministic_output[-1])

		current_minibatch_size = len(data_indices[minibatch_start_index:minibatch_start_index + minibatch_size])
		minibatch_outputs[minibatch_index, 0] = current_minibatch_size
		minibatch_outputs[minibatch_index, 1:5] = function_stochastic_output
		minibatch_outputs[minibatch_index, 5:9] = function_deterministic_output

		minibatch_start_index += minibatch_size
		minibatch_index += 1

	assert numpy.sum(minibatch_outputs[:, 0]) == number_of_data, (numpy.sum(minibatch_outputs[:, 0]), number_of_data)

	if output_file!=None:
		numpy.save(output_file, minibatch_outputs)

	dataset_outputs = numpy.sum(minibatch_outputs[:, 0][:, numpy.newaxis] * minibatch_outputs[:, 1:], axis=0) / number_of_data

	logger.debug("stochastic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
	             (dataset_outputs[1], dataset_outputs[2], dataset_outputs[3], dataset_outputs[0] * 100))
	print("debug: stochastic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
	             (dataset_outputs[1], dataset_outputs[2], dataset_outputs[3], dataset_outputs[0] * 100))

	logger.debug("deterministic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
	             (dataset_outputs[5], dataset_outputs[6], dataset_outputs[7], dataset_outputs[4] * 100))
	print("debug: deterministic: loss %g, objective %g, regularizer %g, accuracy %g%%" %
	             (dataset_outputs[5], dataset_outputs[6], dataset_outputs[7], dataset_outputs[4] * 100))

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
		if isinstance(layer, AdaptiveDropoutLayer):
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
		if isinstance(layer, AdaptiveDropoutLayer):
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
		if isinstance(layer, AdaptiveDropoutLayer):
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


def snapshot_dropout(network, settings=None, **kwargs):
	dropout_layer_index = 0
	for network_layer in network.get_network_layers():
		if isinstance(network_layer, layers.BernoulliDropoutLayer) or \
				isinstance(network_layer, layers.AdaptiveDropoutLayer) or \
				isinstance(network_layer, layers.DynamicDropoutLayer):
			layer_retain_probability = network_layer.activation_probability.eval()
		elif isinstance(network_layer, layers.SparseVariationalDropoutLayer):
			sigma = T.nnet.sigmoid(network_layer.logit_sigma).eval()
			layer_retain_probability = 1. / (1. + sigma ** 2)
		# alpha = T.exp(network_layer.logit_sigma).eval()
		# layer_retain_probability = 1. / (1. + alpha)
		elif isinstance(network_layer, layers.VariationalDropoutTypeALayer) or \
				isinstance(network_layer, layers.VariationalDropoutTypeBLayer) or \
				isinstance(network_layer, layers.SparseVariationalDropoutLayer):
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

def snapshot_dense(network, settings=None, **kwargs):
	dense_layer_index = 0
	for network_layer in network.get_network_layers():
		if isinstance(network_layer, layers.DenseLayer):
			layer_weight = network_layer.W.eval()
			layer_bias = network_layer.b.eval()
		else:
			continue

		if settings is not None:
			# layer_retain_probability = numpy.reshape(layer_retain_probability, numpy.prod(layer_retain_probability.shape))
			dense_file = os.path.join(settings.output_directory, "dense.%d.epoch.%d.npy" % (dense_layer_index, network.epoch_index))
			numpy.save(dense_file, [layer_weight, layer_bias])
		dense_layer_index += 1

def snapshot_convolution(network, settings, **kwargs):
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