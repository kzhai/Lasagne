import logging
import os
import timeit
import types
from itertools import chain

import numpy
import theano
import theano.tensor

from .. import layers
from .. import objectives, regularization, updates, utils, Xpolicy, Xregularization

logger = logging.getLogger(__name__)

__all__ = [
	"adjust_parameter_according_to_policy",
	#
	#
	#
	"Network",
	"FeedForwardNetwork",
	"RecurrentNetwork",
	#
	#
	#
	"GenerativeNetwork",
]


def adjust_parameter_according_to_policy(parameter_policy, epoch_index):
	if parameter_policy[1] is Xpolicy.constant:
		parameter_setting = Xpolicy.constant(parameter_policy[0])
	elif parameter_policy[1] is Xpolicy.piecewise_constant:
		parameter_setting = parameter_policy[1](parameter_policy[0], epoch_index, parameter_policy[2],
		                                        parameter_policy[3])
	elif parameter_policy[1] is Xpolicy.inverse_time_decay \
			or parameter_policy[1] is Xpolicy.natural_exp_decay \
			or parameter_policy[1] is Xpolicy.exponential_decay:
		parameter_setting = parameter_policy[1](parameter_policy[0], epoch_index, parameter_policy[2],
		                                        parameter_policy[3], parameter_policy[4])

	return parameter_setting.astype(theano.config.floatX)


class Network(object):
	def __init__(self,
	             incoming,
	             objective_functions,
	             update_function,
	             learning_rate_policy=[1e-3, Xpolicy.constant],
	             # learning_rate_decay=None,
	             max_norm_constraint=0,
	             ):
		if isinstance(incoming, tuple):
			self._input_shape = incoming
			self._input_layer = layers.InputLayer(shape=self._input_shape)
		else:
			self._input_shape = incoming.output_shape
			self._input_layer = incoming

		if any(d is not None and d <= 0 for d in self._input_shape):
			raise ValueError("Cannot create Layer with a non-positive input_shape dimension. input_shape=%r" %
			                 self._input_shape)

		input_layers = [layer for layer in layers.get_all_layers(self._input_layer) if isinstance(layer,
		                                                                                          layers.InputLayer)]
		assert len(input_layers) == 1

		# self._input_variable = input_layers[0].input_var
		self._input_variable = self._input_layer.input_var
		# self._learning_rate_variable = theano.tensor.scalar()
		self._learning_rate_variable = theano.shared(value=numpy.array(1e-3).astype(theano.config.floatX),
		                                             name="learning_rate")

		#
		#
		#
		#
		#

		self.epoch_index = 0
		self.minibatch_index = 0

		self.objective_functions_change_stack = []
		self.update_function_change_stack = []
		self.learning_rate_policy_change_stack = []

		self.regularizer_functions_change_stack = []
		self.max_norm_constraint_change_stack = []

		self.__set_objective_functions(objective_functions)
		self.__set_update_function(update_function)
		self.set_learning_rate_policy(learning_rate_policy)

		self.__set_regularizer_functions()
		self.set_max_norm_constraint(max_norm_constraint)

	def get_network_input(self, **kwargs):
		return layers.get_output(self._input_layer, **kwargs)

	def get_output(self, inputs=None, **kwargs):
		return layers.get_output(self._neural_network, inputs, **kwargs)

	def get_output_shape(self, input_shapes=None):
		return layers.get_output_shape(self._neural_network, input_shapes)

	def get_network_layers(self):
		return layers.get_all_layers(self._neural_network, [self._input_layer])

	def get_network_params(self, **tags):
		params = chain.from_iterable(l.get_params(**tags) for l in self.get_network_layers()[1:])
		return utils.unique(params)

	'''
	def count_network_params(self, **tags):
		# return lasagne.layers.count_params(self.get_all_layers()[1:], **tags)
		params = self.get_network_params(**tags)
		shapes = [p.get_value().shape for p in params]
		counts = [numpy.prod(shape) for shape in shapes]
		return sum(counts)
	
	def get_network_param_values(self, **tags):
		# return lasagne.layers.get_all_param_values(self.get_all_layers()[1:], **tags)
		params = self.get_network_params(**tags)
		return [p.get_value() for p in params]
	'''

	def set_input_variable(self, input):
		'''This is to establish the computational graph'''
		# self.get_all_layers()[0].input_var = input
		self._input_variable = input

	#
	#
	#
	#
	#

	@property
	def build_functions(self):
		raise NotImplementedError("Not implemented in successor classes!")

	@property
	def get_objectives(self, **kwargs):
		raise NotImplementedError("Not implemented in successor classes!")

	def get_regularizers(self, regularizer_functions=None, **kwargs):
		assert hasattr(self, "_neural_network")
		regularizer = 0

		if regularizer_functions == None:
			regularizer_functions = self._regularizer_functions

		for regularizer_function, regularizer_weight_variable in regularizer_functions.items():
			assert type(regularizer_function) is types.FunctionType
			if regularizer_function in set(
					[Xregularization.rademacher,
					 Xregularization.rademacher_p_2_q_2,
					 Xregularization.rademacher_p_1_q_inf,
					 Xregularization.rademacher_p_inf_q_1,
					 Xregularization.kl_divergence_kingma,
					 Xregularization.kl_divergence_sparse]):
				# assert type(lambda_policy) is float
				# decayed_lambda = decay_parameter(lambda_decay_policy, self.epoch_index)
				# regularizer += decayed_lambda * regularizer_function(self, **kwargs)

				regularizer += regularizer_weight_variable * regularizer_function(self, **kwargs)
			elif regularizer_function in set([regularization.l1, regularization.l2]):
				regularizer += regularizer_weight_variable * regularization.regularize_network_params(
					self._neural_network,
					regularizer_function, **kwargs)
				'''
				if type(lambda_decay_policy) is list:
					dense_layers = []
					for layer in self.get_network_layers():
						if isinstance(layer, layers.dense.DenseLayer):
							dense_layers.append(layer)
					assert len(dense_layers) == len(lambda_decay_policy), (dense_layers, lambda_decay_policy)
					regularizer += regularization.regularize_layer_params_weighted(
						dict(list(zip(dense_layers, lambda_decay_policy))), regularizer_function, **kwargs)
				elif type(lambda_decay_policy) is float:
					decayed_lambda = decay_rate(lambda_decay_policy, self.epoch_index)
					regularizer += decayed_lambda * regularization.regularize_network_params(self._neural_network,
					                                                                  regularizer_function, **kwargs)
				else:
					logger.error(
						"unrecognized regularizer function settings: %s, %s" % (regularizer_function, lambda_decay_policy))
				'''
			else:
				logger.error("unrecognized regularizer function: %s" % (regularizer_function))

		return regularizer

	@property
	def get_loss(self, **kwargs):
		raise NotImplementedError("Not implemented in successor classes!")

	#
	#
	#
	#
	#

	def __set_objective_functions(self, objective_functions):
		assert objective_functions is not None
		if type(objective_functions) is types.FunctionType:
			self._objective_functions = {objective_functions: 1.0}
		elif type(objective_functions) is list:
			self._objective_functions = {objective_function: 1.0 for objective_function in objective_functions}
		else:
			logger.error('unrecognized objective functions: %s' % (objective_functions))
		self.objective_functions_change_stack.append((self.epoch_index, self._objective_functions))

	def set_objectives(self, objectives):
		self.__set_objective_functions(objectives)
		self.build_functions()

	def __set_regularizer_functions(self, regularizer_functions=None):
		_regularizer_functions = {}
		_regularizer_lambda_policy = {};
		if regularizer_functions is not None:
			assert hasattr(self, "_neural_network")
			assert type(regularizer_functions) is dict
			'''
			if type(regularizer_functions) is types.FunctionType:
				regularizer_weight_variable = theano.shared(value=numpy.array(1.0).astype(theano.config.floatX),
				                                            name="%s" % regularizer_functions)

				_regularizer_functions[regularizer_functions] = regularizer_weight_variable
			'''

			for regularizer_function, lambda_decay_policy in regularizer_functions.items():
				assert type(regularizer_function) is types.FunctionType
				assert type(lambda_decay_policy) is list;

				'''
				if type(lambda_decay_policy) is list:
					for weight in lambda_decay_policy:
						# assert isinstance(layer, layers.Layer)
						assert type(weight) == float
				else:
					assert type(lambda_decay_policy) == float
				'''
				regularizer_weight_variable = theano.shared(value=numpy.array(0.).astype(theano.config.floatX),
				                                            name="%s lambda" % regularizer_function)
				_regularizer_functions[regularizer_function] = regularizer_weight_variable
				_regularizer_lambda_policy[regularizer_weight_variable] = lambda_decay_policy

			# _regularizer_functions[regularizer_function] = lambda_decay_policy
		self._regularizer_functions = _regularizer_functions
		self._regularizer_lambda_policy = _regularizer_lambda_policy
		self.regularizer_functions_change_stack.append((self.epoch_index, self._regularizer_functions))

	def set_regularizers(self, regularizers=None):
		self.__set_regularizer_functions(regularizers)
		self.build_functions()

	def __set_update_function(self, update_function):
		assert update_function is not None
		self._update_function = update_function
		self.update_function_change_stack.append((self.epoch_index, self._update_function))

	def set_update(self, update):
		self.__set_update_function(update)
		self.build_functions()

	def set_learning_rate_policy(self, learning_rate_policy):
		self.learning_rate_policy = learning_rate_policy
		self.learning_rate_policy_change_stack.append((self.epoch_index, self.learning_rate_policy))

	def set_max_norm_constraint(self, max_norm_constraint):
		self.max_norm_constraint = max_norm_constraint
		self.max_norm_constraint_change_stack.append((self.epoch_index, self.max_norm_constraint))

	def __update_learning_rate(self):
		self._learning_rate_variable.set_value(
			adjust_parameter_according_to_policy(self.learning_rate_policy, self.epoch_index))

	def __update_regularizer_weight(self):
		if not hasattr(self, "_regularizer_lambda_policy"):
			return

		for regularizer_weight_variable, lambda_decay_policy in self._regularizer_lambda_policy.items():
			regularizer_weight_variable.set_value(
				adjust_parameter_according_to_policy(lambda_decay_policy, self.epoch_index))

	def update_shared_variables(self):
		self.__update_learning_rate()
		self.__update_regularizer_weight()

	def debug(self, settings, **kwargs):
		raise NotImplementedError("Not implemented in successor classes!")


class FeedForwardNetwork(Network):
	def __init__(self, incoming,
	             objective_functions,
	             update_function,
	             learning_rate_policy=[1e-3, Xpolicy.constant],
	             # learning_rate_decay=None,
	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,
	             validation_interval=-1,
	             ):

		super(FeedForwardNetwork, self).__init__(incoming,
		                                         objective_functions,
		                                         update_function,
		                                         learning_rate_policy,
		                                         # learning_rate_decay,
		                                         max_norm_constraint,
		                                         # learning_rate_decay_style,
		                                         # learning_rate_decay_parameter
		                                         )

		self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		self.validation_interval = validation_interval
		self.best_epoch_index = 0
		self.best_minibatch_index = 0
		self.best_validate_accuracy = 0

	def get_objectives(self, label, objective_functions=None, threshold=1e-9, **kwargs):
		output = self.get_output(**kwargs)
		# output = theano.tensor.clip(output, threshold, 1.0 - threshold)
		if objective_functions is None:
			# objective = theano.tensor.mean(self._objective_functions(output, label), dtype=theano.config.floatX)
			objective = 0
			for objective_function, weight in list(self._objective_functions.items()):
				objective += weight * theano.tensor.mean(objective_function(output, label), dtype=theano.config.floatX)
		else:
			# TODO: expand to multiple objective functions
			temp_objective_function = getattr(objectives, objective_functions)
			objective = theano.tensor.mean(temp_objective_function(output, label), dtype=theano.config.floatX)
		return objective

	def get_loss(self, label, **kwargs):
		loss = self.get_objectives(label, **kwargs) + self.get_regularizers()
		return loss

	def build_functions(self):
		# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
		nondeterministic_loss = self.get_loss(self._output_variable)
		nondeterministic_objective = self.get_objectives(self._output_variable)
		nondeterministic_accuracy = self.get_objectives(self._output_variable,
		                                                objective_functions="categorical_accuracy")

		# Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
		deterministic_loss = self.get_loss(self._output_variable, deterministic=True)
		deterministic_objective = self.get_objectives(self._output_variable, deterministic=True)
		deterministic_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy",
		                                             deterministic=True)

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		trainable_params = self.get_network_params(trainable=True)
		trainable_params_nondeterministic_updates = self._update_function(nondeterministic_loss,
		                                                                  trainable_params,
		                                                                  self._learning_rate_variable,
		                                                                  momentum=0.95)

		if self.max_norm_constraint > 0:
			for param in self.get_network_params(trainable=True, regularizable=True):
				ndim = param.ndim
				if ndim == 2:  # DenseLayer
					sum_over = (0,)
				elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
					sum_over = tuple(range(1, ndim))
				elif ndim == 6:  # LocallyConnected{2}DLayer
					sum_over = tuple(range(1, ndim))
				else:
					continue
				# raise ValueError("Unsupported tensor dimensionality {}.".format(ndim))
				trainable_params_nondeterministic_updates[param] = updates.norm_constraint(
					trainable_params_nondeterministic_updates[param],
					self.max_norm_constraint,
					norm_axes=sum_over)

		# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
		self._function_train_trainable_params_nondeterministic = theano.function(
			# inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			inputs=[self._input_variable, self._output_variable],
			outputs=[nondeterministic_objective, nondeterministic_accuracy],
			updates=trainable_params_nondeterministic_updates,
			on_unused_input='warn'
		)

		# Compile a second function computing the validation train_loss and accuracy:
		self._function_test = theano.function(
			inputs=[self._input_variable, self._output_variable],
			outputs=[deterministic_objective, deterministic_accuracy],
			on_unused_input='warn'
		)

		'''
		# TODO: begin{this is only for adaptive dropout layers}
		adaptable_params = self.get_network_params(adaptable=True)
		adaptable_params_deterministic_updates = self._update_function(deterministic_loss, adaptable_params,
		                                                               self._learning_rate_variable)

		# Compile a second function computing the validation train_loss and accuracy:
		self._function_train_adaptable_params_deterministic = theano.function(
			inputs=[self._input_variable, self._output_variable],
			outputs=[deterministic_objective, deterministic_accuracy],
			updates=adaptable_params_deterministic_updates,
			on_unused_input='warn'
		)
		# TODO: end{this is only for adaptive dropout layers}
		'''

		'''
		self._function_debugger = theano.function(
			inputs=[self._input_variable, self._output_variable],
			outputs=[nondeterministic_loss, nondeterministic_objective, deterministic_loss, deterministic_objective],
		)
		'''

	def test(self, test_dataset):
		test_dataset_x, test_dataset_y = test_dataset
		test_running_time = timeit.default_timer()
		test_function_outputs = self._function_test(test_dataset_x, test_dataset_y)
		average_test_objective = test_function_outputs[0]
		average_test_accuracy = test_function_outputs[1]
		test_running_time = timeit.default_timer() - test_running_time
		logger.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, test_running_time, average_test_objective,
			average_test_accuracy * 100))
		print('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, test_running_time, average_test_objective,
			average_test_accuracy * 100))

	def validate(self, validate_dataset, test_dataset=None, best_model_file_path=None):
		validate_running_time = timeit.default_timer()
		validate_dataset_x, validate_dataset_y = validate_dataset
		validate_function_outputs = self._function_test(validate_dataset_x, validate_dataset_y)
		average_validate_objective = validate_function_outputs[0]
		average_validate_accuracy = validate_function_outputs[1]
		validate_running_time = timeit.default_timer() - validate_running_time
		logger.info('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, validate_running_time, average_validate_objective,
			average_validate_accuracy * 100))
		print('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, validate_running_time, average_validate_objective,
			average_validate_accuracy * 100))

		# if we got the best validation score until now
		if average_validate_accuracy > self.best_validate_accuracy:
			self.best_epoch_index = self.epoch_index
			self.best_minibatch_index = self.minibatch_index
			self.best_validate_accuracy = average_validate_accuracy
			# self.best_validate_model = copy.deepcopy(self)

			if best_model_file_path is not None:
				# save the best model
				# cPickle.dump(self, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
				logger.info('\tbest model found: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
					self.epoch_index, self.minibatch_index, average_validate_objective,
					average_validate_accuracy * 100))

		if test_dataset is not None:
			self.test(test_dataset)

	def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None):
		# In each epoch_index, we do a full pass over the training data:
		epoch_running_time = 0

		train_dataset_x, train_dataset_y = train_dataset

		number_of_data = train_dataset_x.shape[0]
		data_indices = numpy.random.permutation(number_of_data)
		minibatch_start_index = 0

		self.update_shared_variables()

		total_train_objective = 0
		total_train_accuracy = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data
			minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]
			minibatch_start_index += minibatch_size

			minibatch_x = train_dataset_x[minibatch_indices, :]
			minibatch_y = train_dataset_y[minibatch_indices]

			minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy = self.train_minibatch(
				minibatch_x, minibatch_y)

			epoch_running_time += minibatch_running_time

			current_minibatch_size = len(data_indices[minibatch_start_index:minibatch_start_index + minibatch_size])
			total_train_objective += minibatch_average_train_objective * current_minibatch_size
			total_train_accuracy += minibatch_average_train_accuracy * current_minibatch_size

			# And a full pass over the validation data:
			if validate_dataset is not None and self.validation_interval > 0 and self.minibatch_index % self.validation_interval == 0:
				average_train_accuracy = total_train_accuracy / number_of_data
				average_train_objective = total_train_objective / number_of_data
				logger.info('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
					self.epoch_index, self.minibatch_index, average_train_objective, average_train_accuracy * 100))

				output_file = None
				if output_directory is not None:
					output_file = os.path.join(output_directory, 'model.pkl')
				self.validate(validate_dataset, test_dataset, output_file)

			self.minibatch_index += 1

		if validate_dataset is not None:
			output_file = None
			if output_directory is not None:
				output_file = os.path.join(output_directory, 'model.pkl')
			self.validate(validate_dataset, test_dataset, output_file)
		elif test_dataset is not None:
			# if output_directory != None:
			# output_file = os.path.join(output_directory, 'model-%d.pkl' % self.epoch_index)
			# cPickle.dump(self, open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
			self.test(test_dataset)

		average_train_accuracy = total_train_accuracy / number_of_data
		average_train_objective = total_train_objective / number_of_data
		logger.info('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, epoch_running_time, average_train_objective,
			average_train_accuracy * 100))
		print('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, epoch_running_time, average_train_objective,
			average_train_accuracy * 100))

		return epoch_running_time

	def train_minibatch(self, minibatch_x, minibatch_y):
		minibatch_running_time = timeit.default_timer()
		train_trainable_params_function_outputs = self._function_train_trainable_params_nondeterministic(minibatch_x,
		                                                                                                 minibatch_y)
		minibatch_average_train_objective, minibatch_average_train_accuracy = train_trainable_params_function_outputs

		'''
		# TODO: begin{this is only for adaptive dropout layers}
		train_adaptable_params_function_outputs = self._function_train_adaptable_params_deterministic(minibatch_x,
		                                                                                              minibatch_y)
		# TODO: end{this is only for adaptive dropout layers}
		'''

		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		return minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy


def parse_sequence(dataset, window_size, sequence_length, position_offset=-1):
	if dataset is None:
		return None

	sequence_set_x, sequence_set_y = dataset
	# Parse data into sequences
	sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32)
	sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8)
	sequence_y = numpy.zeros(0, dtype=numpy.int32)

	sequence_indices_by_instance = [0]
	for instance_x, instance_y in zip(sequence_set_x, sequence_set_y):
		# context_windows = get_context_windows(train_sequence_x, window_size)
		# train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, backprop_step)
		instance_sequence_x, instance_sequence_m = get_context_sequences(instance_x, sequence_length, window_size,
		                                                                 position_offset)
		assert len(instance_sequence_x) == len(instance_sequence_m)
		assert len(instance_sequence_x) == len(instance_y)

		sequence_x = numpy.concatenate((sequence_x, instance_sequence_x), axis=0)
		sequence_m = numpy.concatenate((sequence_m, instance_sequence_m), axis=0)
		sequence_y = numpy.concatenate((sequence_y, instance_y), axis=0)

		sequence_indices_by_instance.append(len(sequence_y))

	return sequence_x, sequence_y, sequence_m, sequence_indices_by_instance


def get_context_sequences(instance, sequence_length, window_size, position_offset=-1):
	'''
	context_windows :: list of word idxs
	return a list of minibatches of indexes
	which size is equal to backprop_step
	border cases are treated as follow:
	eg: [0,1,2,3] and backprop_step = 3
	will output:
	[[0],[0,1],[0,1,2],[1,2,3]]
	'''

	context_windows = get_context(instance, window_size, position_offset)
	sequences_x, sequences_m = get_sequences(context_windows, sequence_length)
	return sequences_x, sequences_m


def get_context(instance, window_size, position_offset=-1, vocab_size=None):
	'''
	window_size :: int corresponding to the size of the window
	given a list of indexes composing a sentence
	it will return a list of list of indexes corresponding
	to context windows surrounding each word in the sentence
	'''

	assert window_size >= 1
	if position_offset < 0:
		assert window_size % 2 == 1
		position_offset = window_size / 2
	assert position_offset < window_size

	instance = list(instance)

	if vocab_size is None:
		context_windows = -numpy.ones((len(instance), window_size), dtype=numpy.int32)
		# padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
		padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
		for i in xrange(len(instance)):
			context_windows[i, :] = padded_sequence[i:i + window_size]
	else:
		context_windows = numpy.zeros((len(instance), vocab_size), dtype=numpy.int32)
		# padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
		padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
		for i in xrange(len(instance)):
			for j in padded_sequence[i:i + window_size]:
				context_windows[i, j] += 1

	# assert len(context_windows) == len(sequence)
	return context_windows


def get_sequences(context_windows, sequence_length):
	'''
	context_windows :: list of word idxs
	return a list of minibatches of indexes
	which size is equal to backprop_step
	border cases are treated as follow:
	eg: [0,1,2,3] and backprop_step = 3
	will output:
	[[0],[0,1],[0,1,2],[1,2,3]]
	'''

	number_of_tokens, window_size = context_windows.shape
	sequences_x = -numpy.ones((number_of_tokens, sequence_length, window_size), dtype=numpy.int32)
	sequences_m = numpy.zeros((number_of_tokens, sequence_length), dtype=numpy.int32)
	for i in xrange(min(number_of_tokens, sequence_length)):
		sequences_x[i, 0:i + 1, :] = context_windows[0:i + 1, :]
		sequences_m[i, 0:i + 1] = 1
	for i in xrange(min(number_of_tokens, sequence_length), number_of_tokens):
		sequences_x[i, :, :] = context_windows[i - sequence_length + 1:i + 1, :]
		sequences_m[i, :] = 1
	return sequences_x, sequences_m


class RecurrentNetwork(FeedForwardNetwork):
	def __init__(self,
	             # incoming,
	             # incoming_mask,
	             sequence_length,
	             # recurrent_type,

	             objective_functions,
	             update_function,
	             learning_rate_policy=1e-3,
	             # learning_rate_decay=None,

	             max_norm_constraint=0,
	             total_norm_constraint=0,
	             normalize_embeddings=False,
	             validation_interval=-1,

	             window_size=1,
	             position_offset=0,
	             # gradient_steps=-1,
	             # gradient_clipping=0,
	             ):

		self._sequence_length = sequence_length

		self._window_size = window_size
		self._position_offset = position_offset

		# self._recurrent_type = recurrent_type
		# self._gradient_steps = gradient_steps
		# self._gradient_clipping = gradient_clipping

		incoming = (None, sequence_length, window_size)
		incoming_mask = (None, sequence_length)

		if isinstance(incoming, tuple):
			incoming = layers.InputLayer(shape=incoming, input_var=theano.tensor.itensor3())

		super(RecurrentNetwork, self).__init__(incoming,
		                                       objective_functions,
		                                       update_function,
		                                       learning_rate_policy,
		                                       # learning_rate_decay,
		                                       max_norm_constraint,
		                                       validation_interval
		                                       )

		self.total_norm_constraint = total_norm_constraint
		self.normalize_embeddings = normalize_embeddings

		if isinstance(incoming_mask, tuple):
			self._input_mask_shape = incoming_mask
			self._input_mask_layer = layers.InputLayer(shape=incoming_mask, input_var=theano.tensor.imatrix())
		else:
			self._input_mask_shape = incoming_mask.output_shape
			self._input_mask_layer = incoming_mask

		if any(d is not None and d <= 0 for d in self._input_mask_shape):
			raise ValueError("Cannot create Layer with a non-positive input_shape dimension. input_shape=%r" %
			                 self._input_mask_shape)

		self._input_mask_variable = self._input_mask_layer.input_var

	def parse_sequence(self, dataset):
		if dataset is None:
			return None
		return parse_sequence(dataset, self._window_size, self._sequence_length, self._position_offset)

	def build_functions(self):
		# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
		train_loss = self.get_loss(self._output_variable)
		train_objective = self.get_objectives(self._output_variable)
		train_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy")

		# Create update expressions for training, i.e., how to modify the parameters at each training step.
		# Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		trainable_params = self.get_network_params(trainable=True)
		if self.total_norm_constraint > 0:
			trainable_grads = theano.tensor.grad(train_loss, trainable_params)
			scaled_trainable_grads = updates.total_norm_constraint(trainable_grads, self.total_norm_constraint)
			trainable_params_updates = self._update_function(scaled_trainable_grads, trainable_params,
			                                                 self._learning_rate_variable)
		else:
			trainable_params_updates = self._update_function(train_loss, trainable_params, self._learning_rate_variable)

		if self.max_norm_constraint > 0:
			for param in self.get_network_params(trainable=True):
				ndim = param.ndim
				if ndim == 2:  # DenseLayer
					sum_over = (0,)
				elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
					sum_over = tuple(range(1, ndim))
				elif ndim == 6:  # LocallyConnected{2}DLayer
					sum_over = tuple(range(1, ndim))
				else:
					continue
				# raise ValueError("Unsupported tensor dimensionality {} of param {}.".format(ndim, param))

				trainable_params_updates[param] = updates.norm_constraint(trainable_params_updates[param],
				                                                          self.max_norm_constraint,
				                                                          norm_axes=sum_over)

		# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
		self._train_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable,
			        self._learning_rate_variable],
			outputs=[train_objective, train_accuracy],
			updates=trainable_params_updates
		)

		# Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
		test_loss = self.get_loss(self._output_variable, deterministic=True)
		test_objective = self.get_objectives(self._output_variable, deterministic=True)
		test_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy",
		                                    deterministic=True)
		# As a bonus, also create an expression for the classification accuracy:
		# test_prediction = self.get_output(deterministic=True)
		# test_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

		# Compile a second function computing the validation train_loss and accuracy:
		self._test_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable],
			outputs=[test_objective, test_accuracy],
		)

		if self.normalize_embeddings:
			# Compile a function to normalize all the embeddings
			self._embeddings = self._embedding_layer.W
			self._normalize_embeddings_function = theano.function(
				inputs=[],
				updates={
					self._embeddings: self._embeddings / theano.tensor.sqrt(
						(self._embeddings ** 2).sum(axis=1)).dimshuffle(
						0, 'x')}
			)

		'''
		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable,
		                                       objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		adaptable_params_updates = self._update_function(dropout_loss, adaptable_params, self._learning_rate_variable)

		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs=[dropout_loss, dropout_accuracy],
			updates=adaptable_params_updates
		)
		'''

		'''
		from debugger import debug_rademacher
		self._debug_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs = debug_rademacher(self, self._output_variable, deterministic=True),
			#outputs=[self.get_objectives(self._output_variable, determininistic=True), self.get_loss(self._output_variable, deterministic=True)],
			on_unused_input='ignore'
		)
		'''

	def test(self, test_dataset):
		test_dataset_x, test_dataset_y, test_dataset_m, test_sequence_indices_by_instance = test_dataset
		test_running_time = timeit.default_timer()
		test_function_outputs = self._test_function(test_dataset_x, test_dataset_y, test_dataset_m)
		average_test_loss = test_function_outputs[0]
		average_test_accuracy = test_function_outputs[1]
		test_running_time = timeit.default_timer() - test_running_time
		logger.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
			average_test_accuracy * 100))
		print('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
			average_test_accuracy * 100))

	def validate(self, validate_dataset, test_dataset=None, best_model_file_path=None):
		validate_running_time = timeit.default_timer()
		validate_dataset_x, validate_dataset_y, validate_dataset_m, validate_sequence_indices_by_instance = validate_dataset
		validate_function_outputs = self._test_function(validate_dataset_x, validate_dataset_y, validate_dataset_m)
		average_validate_loss = validate_function_outputs[0]
		average_validate_accuracy = validate_function_outputs[1]
		validate_running_time = timeit.default_timer() - validate_running_time
		logger.info('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
			average_validate_accuracy * 100))
		print('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
			average_validate_accuracy * 100))

		# if we got the best validation score until now
		if average_validate_accuracy > self.best_validate_accuracy:
			self.best_epoch_index = self.epoch_index
			self.best_minibatch_index = self.minibatch_index
			self.best_validate_accuracy = average_validate_accuracy

			if best_model_file_path is not None:
				# save the best model
				# cPickle.dump(self, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
				logger.info('\tbest model found: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
					self.epoch_index, self.minibatch_index, average_validate_loss, average_validate_accuracy * 100))

		if test_dataset is not None:
			self.test(test_dataset)
			'''
			test_running_time = timeit.default_timer()
			test_dataset_x, test_dataset_y = test_dataset
			average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y)
			test_running_time = timeit.default_timer() - test_running_time
			logger.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
				self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
				average_test_accuracy * 100))
			'''

	def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None,
	          minibatch_by_instance=True):
		# In each epoch_index, we do a full pass over the training data:
		epoch_running_time = 0

		train_sequence_x, train_sequence_y, train_sequence_m, train_sequence_indices_by_instance = train_dataset

		if minibatch_by_instance:
			number_of_data = len(train_sequence_indices_by_instance) - 1
		else:
			number_of_data = len(train_sequence_x)
		# data_indices = numpy.random.permutation(number_of_data)

		self.update_shared_variables()

		minibatch_start_index = 0
		'''
		if self.learning_rate_decay[0] == "epoch":
			learning_rate = decay_learning_rate(self.learning_rate, self.epoch_index, self.learning_rate_decay)
		'''

		total_train_objective = 0
		total_train_accuracy = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data

			sequence_start_index = minibatch_start_index
			sequence_end_index = min(minibatch_start_index + minibatch_size, number_of_data)
			# print("checkpoint a", sequence_start_index, sequence_end_index, number_of_data)

			if minibatch_by_instance:
				sequence_start_index = train_sequence_indices_by_instance[sequence_start_index]
				sequence_end_index = train_sequence_indices_by_instance[sequence_end_index]
				'''
				minibatch_indices = data_indices[sequence_start_index:sequence_end_index]
				minibatch_x = train_sequence_x[minibatch_indices]
				minibatch_y = train_sequence_y[minibatch_indices]
				minibatch_m = train_sequence_m[minibatch_indices]
				'''

			# print("checkpoint b", sequence_start_index, sequence_end_index, number_of_data)

			minibatch_x = train_sequence_x[sequence_start_index:sequence_end_index]
			minibatch_y = train_sequence_y[sequence_start_index:sequence_end_index]
			minibatch_m = train_sequence_m[sequence_start_index:sequence_end_index]

			minibatch_start_index += minibatch_size

			minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy = self.train_minibatch(
				minibatch_x, minibatch_y, minibatch_m)
			# print('minibatch %i, loss %f, accuracy %f%%' % (self.minibatch_index, minibatch_average_train_objective, minibatch_average_train_accuracy * 100))

			'''
			if minibatch_average_train_objective>10:
				minibatch_running_time = timeit.default_timer()
				print(self._debug_function(minibatch_x, minibatch_y, learning_rate))

				train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
				minibatch_average_train_objective, minibatch_average_train_accuracy = train_function_outputs
				minibatch_running_time = timeit.default_timer() - minibatch_running_time
			'''

			epoch_running_time += minibatch_running_time

			current_minibatch_size = sequence_end_index - sequence_start_index
			total_train_objective += minibatch_average_train_objective * current_minibatch_size
			total_train_accuracy += minibatch_average_train_accuracy * current_minibatch_size

			# And a full pass over the validation data:
			if validate_dataset is not None and self.validation_interval > 0 and self.minibatch_index % self.validation_interval == 0:
				average_train_accuracy = total_train_accuracy / sequence_end_index
				average_train_objective = total_train_objective / sequence_end_index
				logger.info('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
					self.epoch_index, self.minibatch_index, average_train_objective, average_train_accuracy * 100))

				output_file = None
				if output_directory is not None:
					output_file = os.path.join(output_directory, 'model.pkl')
				self.validate(validate_dataset, test_dataset, output_file)

			self.minibatch_index += 1

		if validate_dataset is not None:
			output_file = None
			if output_directory is not None:
				output_file = os.path.join(output_directory, 'model.pkl')
			self.validate(validate_dataset, test_dataset, output_file)
		elif test_dataset is not None:
			# if output_directory != None:
			# output_file = os.path.join(output_directory, 'model-%d.pkl' % self.epoch_index)
			# cPickle.dump(self, open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
			self.test(test_dataset)

		average_train_accuracy = total_train_accuracy / sequence_end_index
		average_train_objective = total_train_objective / sequence_end_index
		logger.info('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, epoch_running_time, average_train_objective,
			average_train_accuracy * 100))
		print('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, epoch_running_time, average_train_objective,
			average_train_accuracy * 100))

		return epoch_running_time

	def train_minibatch(self, minibatch_x, minibatch_y, minibatch_m):
		'''
		learning_rate = self.learning_rate
		if self.learning_rate_decay is not None:
			if self.learning_rate_decay[0] == "epoch":
				learning_rate = decay_learning_rate(self.learning_rate, self.epoch_index, self.learning_rate_decay)
			elif self.learning_rate_decay[0] == "iteration":
				learning_rate = decay_learning_rate(self.learning_rate, self.minibatch_index, self.learning_rate_decay)
		'''

		learning_rate = adjust_parameter_according_to_policy(self.learning_rate_policy, self.epoch_index)

		minibatch_running_time = timeit.default_timer()
		train_function_outputs = self._train_function(minibatch_x, minibatch_y, minibatch_m, learning_rate)
		minibatch_average_train_objective, minibatch_average_train_accuracy = train_function_outputs
		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		if self.normalize_embeddings:
			self._normalize_embeddings_function()
		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy


#
#
#
#
#

class GenerativeNetwork(Network):
	def __init__(self, incoming,
	             objective_functions,
	             update_function,
	             learning_rate_policy=1e-3,
	             learning_rate_decay_style=None,
	             learning_rate_decay_parameter=0,
	             ):
		super(GenerativeNetwork, self).__init__(incoming,
		                                        objective_functions,
		                                        update_function,
		                                        learning_rate_policy,
		                                        learning_rate_decay_style,
		                                        learning_rate_decay_parameter,
		                                        )

	'''
	@property
	def build_functions(self):
		raise NotImplementedError("Not implemented in successor classes!")
	'''

	def get_objectives(self, objective_function=objectives.squared_error, **kwargs):
		objective = theano.tensor.mean(
			theano.tensor.sum(objective_function(self.get_output(**kwargs), self.get_network_input()), axis=1),
			dtype=theano.config.floatX)
		return objective

	def get_loss(self, **kwargs):
		loss = self.get_objectives(**kwargs)

		# loss += self.L1_regularizer()
		# loss += self.L2_regularizer()

		return loss

	def build_functions(self):
		# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
		train_loss = self.get_loss()
		# train_accuracy = self.get_objective(self._output_variable, objective_function="categorical_accuracy")
		# train_prediction = self.get_output(**kwargs)
		# train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		all_params = self.get_network_params(trainable=True)
		all_params_updates = self._update_function(train_loss, all_params, self._learning_rate_variable)

		# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
		self._train_function = theano.function(
			inputs=[self._input_variable, self._learning_rate_variable],
			outputs=[train_loss],
			updates=all_params_updates
		)

	'''
	def test(self, test_dataset):
		test_dataset_x, test_dataset_y = test_dataset
		test_running_time = timeit.default_timer()
		average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y)
		test_running_time = timeit.default_timer() - test_running_time
		return average_test_loss, average_test_accuracy, test_running_time

	def train(self, train_dataset, learning_rate):
		train_dataset_x, train_dataset_y = train_dataset
		train_running_time = timeit.default_timer()
		average_train_loss, average_train_accuracy = self._train_function(train_dataset_x, train_dataset_y, learning_rate)
		train_running_time = timeit.default_timer() - train_running_time
		return average_train_loss, average_train_accuracy, train_running_time
	'''

	'''
	def test(self, test_dataset):
		test_dataset_x, test_dataset_y = test_dataset
		test_running_time = timeit.default_timer()
		average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y)
		test_running_time = timeit.default_timer() - test_running_time
		# average_test_loss, average_test_accuracy, test_running_time = self.network.test(test_dataset)
		logger.info('\t\ttest: epoch %i, minibatch %i, duration %f, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, test_running_time, average_test_loss, average_test_accuracy * 100))
	'''

	'''
	def validate(self, validate_dataset, test_dataset=None, best_model_file_path=None):
		# average_validate_loss, average_validate_accuracy, validate_running_time = self.test(validate_dataset)
		validate_running_time = timeit.default_timer()
		validate_dataset_x, validate_dataset_y = validate_dataset
		average_validate_loss, average_validate_accuracy = self._test_function(validate_dataset_x, validate_dataset_y)
		validate_running_time = timeit.default_timer() - validate_running_time
		logger.info('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
			self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
			average_validate_accuracy * 100))

		# if we got the best validation score until now
		if average_validate_accuracy > self.best_validate_accuracy:
			self.best_epoch_index = self.epoch_index
			self.best_minibatch_index = self.minibatch_index
			self.best_validate_accuracy = average_validate_accuracy
			# self.best_validate_model = copy.deepcopy(self)

			if best_model_file_path is not None:
				# save the best model
				cPickle.dump(self, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
				logger.info('\tbest model found: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
					self.epoch_index, self.minibatch_index, average_validate_loss, average_validate_accuracy * 100))

		if test_dataset is not None:
			test_running_time = timeit.default_timer()
			test_dataset_x, test_dataset_y = test_dataset
			average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y)
			test_running_time = timeit.default_timer() - test_running_time
			logger.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
				self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
				average_test_accuracy * 100))
	'''

	def train(self, train_dataset, minibatch_size, output_directory=None):
		# In each epoch_index, we do a full pass over the training data:
		epoch_running_time = 0

		number_of_data = train_dataset.shape[0]
		data_indices = numpy.random.permutation(number_of_data)
		minibatch_start_index = 0

		total_train_loss = 0
		# total_train_accuracy = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data
			minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]
			minibatch_start_index += minibatch_size

			minibatch_x = train_dataset[minibatch_indices, :]
			# minibatch_y = train_dataset_y[minibatch_indices]

			learning_rate = adjust_parameter_according_to_policy(self.minibatch_index, self.learning_rate_policy,
			                                                     self.learning_rate_decay_style,
			                                                     self.learning_rate_decay_parameter)

			minibatch_running_time = timeit.default_timer()
			minibatch_average_train_loss, = self._train_function(minibatch_x, learning_rate)
			minibatch_running_time = timeit.default_timer() - minibatch_running_time
			epoch_running_time += minibatch_running_time

			current_minibatch_size = len(data_indices[minibatch_start_index:minibatch_start_index + minibatch_size])
			total_train_loss += minibatch_average_train_loss * current_minibatch_size
			# total_train_accuracy += minibatch_average_train_accuracy * current_minibatch_size

			# average_train_accuracy = total_train_accuracy / number_of_data
			# average_train_loss = total_train_loss / number_of_data
			# logger.debug('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
			# self.epoch_index, self.minibatch_index, average_train_loss, average_train_accuracy * 100))

			self.minibatch_index += 1

		# average_train_accuracy = total_train_accuracy / number_of_data
		average_train_loss = total_train_loss / number_of_data
		logger.info('train: epoch %i, minibatch %i, duration %fs, loss %f' % (
			self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss))

		# print 'train: epoch %i, minibatch %i, duration %fs, loss %f' % (
		# self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss)

		# self.epoch_index += 1

		return epoch_running_time
