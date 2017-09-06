import logging
import timeit

import numpy
import theano
import theano.tensor

from . import FeedForwardNetwork, RecurrentNetwork, adjust_parameter_according_to_policy
from .. import updates

logger = logging.getLogger(__name__)

__all__ = [
	"DynamicFeedForwardNetwork",
	"ElasticFeedForwardNetwork",
	"DynamicRecurrentNetwork",
]


class DynamicFeedForwardNetwork(FeedForwardNetwork):
	def __init__(self,
	             incoming,
	             objective_functions,
	             update_function,
	             learning_rate_policy=1e-3,
	             # learning_rate_decay=None,

	             dropout_learning_rate_policy=1e-3,
	             # dropout_learning_rate_decay=None,
	             dropout_rate_update_interval=0,

	             max_norm_constraint=0,
	             validation_interval=-1,
	             ):
		super(DynamicFeedForwardNetwork, self).__init__(incoming,
		                                                objective_functions,
		                                                update_function,
		                                                learning_rate_policy,
		                                                # learning_rate_decay,
		                                                max_norm_constraint,
		                                                validation_interval,
		                                                )
		# self._dropout_learning_rate_variable = theano.tensor.scalar()
		self._dropout_learning_rate_variable = theano.shared(value=numpy.array(1e-3).astype(theano.config.floatX),
		                                                     name="dropout_learning_rate")

		self._dropout_rate_update_interval = dropout_rate_update_interval

		# self.dropout_learning_rate_decay_change_stack = []
		self.dropout_learning_rate_change_stack = []

		# self.set_dropout_learning_rate_decay(dropout_learning_rate_decay)
		self.set_dropout_learning_rate_policy(dropout_learning_rate_policy)

	'''
	def set_dropout_learning_rate_decay(self, dropout_learning_rate_decay):
		self.dropout_learning_rate_decay = dropout_learning_rate_decay
		self.dropout_learning_rate_decay_change_stack.append((self.epoch_index, self.dropout_learning_rate_decay))
	'''

	def set_dropout_learning_rate_policy(self, dropout_learning_rate):
		self.dropout_learning_rate_policy = dropout_learning_rate
		self.dropout_learning_rate_change_stack.append((self.epoch_index, self.dropout_learning_rate_policy))

	def __update_dropout_learning_rate(self):
		self._dropout_learning_rate_variable.set_value(
			adjust_parameter_according_to_policy(self.dropout_learning_rate_policy, self.epoch_index))

	def update_shared_variables(self):
		super(DynamicFeedForwardNetwork, self).update_shared_variables();
		self.__update_dropout_learning_rate()

	def build_functions(self):
		super(DynamicFeedForwardNetwork, self).build_functions()

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_objective = self.get_objectives(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable,
		                                       objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		# adaptable_params = self.get_network_params(name="adaptive.dropout.r");
		adaptable_params_updates = self._update_function(dropout_loss, adaptable_params,
		                                                 self._dropout_learning_rate_variable)

		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable],
			outputs=[dropout_objective, dropout_accuracy],
			updates=adaptable_params_updates,
			on_unused_input='warn'
		)

		'''
		from debugger import debug_rademacher
		self._debug_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs = debug_rademacher(self, self._output_variable, deterministic=True),
			#outputs=[self.get_objectives(self._output_variable, determininistic=True), self.get_loss(self._output_variable, deterministic=True)],
			on_unused_input='ignore'
		)
		'''

	def train_minibatch(self, minibatch_x, minibatch_y):
		# minibatch_running_time = timeit.default_timer()
		# train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
		# minibatch_average_train_objective = train_function_outputs[0]
		# minibatch_average_train_accuracy = train_function_outputs[1]
		minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy = super(
			DynamicFeedForwardNetwork, self).train_minibatch(minibatch_x, minibatch_y)

		minibatch_running_time_temp = timeit.default_timer()
		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			# dropout_learning_rate = decay_parameter(self.dropout_learning_rate_policy, self.epoch_index)
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y)

		minibatch_running_time_temp = timeit.default_timer() - minibatch_running_time_temp
		minibatch_running_time += minibatch_running_time_temp
		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy

class ElasticFeedForwardNetwork(FeedForwardNetwork):
	def __init__(self,
	             incoming,
	             objective_functions,
	             update_function,
	             learning_rate_policy=1e-3,
	             # learning_rate_decay=None,

	             prune_policy=[1e-2, 1, 10],
	             #
	             #
	             #
	             #dropout_learning_rate_policy=1e-3,
	             #dropout_rate_update_interval=0,
	             #
	             #
	             #

	             max_norm_constraint=0,
	             validation_interval=-1,
	             ):
		super(ElasticFeedForwardNetwork, self).__init__(incoming,
		                                                objective_functions,
		                                                update_function,
		                                                learning_rate_policy,
		                                                # learning_rate_decay,
		                                                max_norm_constraint,
		                                                validation_interval,
		                                                )

		self._prune_policy = prune_policy

		'''
		# self._dropout_learning_rate_variable = theano.tensor.scalar()
		self._dropout_learning_rate_variable = theano.shared(value=numpy.array(1e-3).astype(theano.config.floatX),
		                                                     name="dropout_learning_rate")

		self._dropout_rate_update_interval = dropout_rate_update_interval

		# self.dropout_learning_rate_decay_change_stack = []
		self.dropout_learning_rate_change_stack = []

		# self.set_dropout_learning_rate_decay(dropout_learning_rate_decay)
		self.set_dropout_learning_rate_policy(dropout_learning_rate_policy)
		'''

	'''
	def build_functions(self):
		super(ElasticFeedForwardNetwork, self).build_functions()

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_objective = self.get_objectives(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable,
		                                       objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		# adaptable_params = self.get_network_params(name="adaptive.dropout.r");
		adaptable_params_updates = self._update_function(dropout_loss, adaptable_params,
		                                                 self._dropout_learning_rate_variable)
		
		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable],
			outputs=[dropout_objective, dropout_accuracy],
			updates=adaptable_params_updates,
			on_unused_input='warn'
		)
	'''

	'''
	def set_dropout_learning_rate_policy(self, dropout_learning_rate):
		self.dropout_learning_rate_policy = dropout_learning_rate
		self.dropout_learning_rate_change_stack.append((self.epoch_index, self.dropout_learning_rate_policy))

	def __update_dropout_learning_rate(self):
		self._dropout_learning_rate_variable.set_value(
			decay_parameter(self.dropout_learning_rate_policy, self.epoch_index))

	def update_shared_variables(self):
		super(DynamicFeedForwardNetwork, self).update_shared_variables();
		self.__update_dropout_learning_rate()
	
	def train_minibatch(self, minibatch_x, minibatch_y):
		# minibatch_running_time = timeit.default_timer()
		# train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
		# minibatch_average_train_objective = train_function_outputs[0]
		# minibatch_average_train_accuracy = train_function_outputs[1]
		minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy = super(
			DynamicFeedForwardNetwork, self).train_minibatch(minibatch_x, minibatch_y)

		minibatch_running_time_temp = timeit.default_timer()
		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			# dropout_learning_rate = decay_parameter(self.dropout_learning_rate_policy, self.epoch_index)
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y)

		minibatch_running_time_temp = timeit.default_timer() - minibatch_running_time_temp
		minibatch_running_time += minibatch_running_time_temp
		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy
	'''

	def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None):
		epoch_running_time = super(ElasticFeedForwardNetwork, self).train(train_dataset, minibatch_size,
		                                                                     validate_dataset, test_dataset,
		                                                                     output_directory)

		#if self.epoch_index >= self._prune_policy[2] and self.epoch_index % self._prune_policy[1] == 0:
		epoch_running_time_temp = timeit.default_timer()
		self.adjust_network(train_dataset, validate_dataset, test_dataset)
		epoch_running_time_temp = timeit.default_timer() - epoch_running_time_temp
		epoch_running_time += epoch_running_time_temp

		return epoch_running_time

	def adjust_network(self, train_dataset=None, validate_dataset=None, test_dataset=None):
		raise NotImplementedError("Not implemented in successor classes!")

class DynamicRecurrentNetwork(RecurrentNetwork):
	def __init__(self,
	             # incoming,
	             # incoming_mask,
	             sequence_length,
	             # recurrent_type,

	             objective_functions,
	             update_function,
	             learning_rate_policy=1e-3,
	             # learning_rate_decay=None,

	             dropout_learning_rate=1e-3,
	             # dropout_learning_rate_decay=None,
	             dropout_rate_update_interval=0,

	             max_norm_constraint=0,
	             total_norm_constraint=0,
	             normalize_embeddings=False,

	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,
	             validation_interval=-1,

	             window_size=1,
	             position_offset=0,
	             # gradient_steps=-1,
	             # gradient_clipping=0,
	             ):
		super(DynamicRecurrentNetwork, self).__init__(
			sequence_length,
			# recurrent_type,

			objective_functions,
			update_function,
			learning_rate_policy,
			# learning_rate_decay,

			max_norm_constraint,
			total_norm_constraint,
			normalize_embeddings,

			validation_interval,

			window_size,
			position_offset,
		)
		self._dropout_learning_rate_variable = theano.tensor.scalar()

		self._dropout_rate_update_interval = dropout_rate_update_interval

		self.dropout_learning_rate_change_stack = []
		self.set_dropout_learning_rate(dropout_learning_rate)

	'''
	def set_dropout_learning_rate_decay(self, dropout_learning_rate_decay):
		self.dropout_learning_rate_decay = dropout_learning_rate_decay
		self.dropout_learning_rate_decay_change_stack.append((self.epoch_index, self.dropout_learning_rate_decay))
	'''

	def set_dropout_learning_rate(self, dropout_learning_rate):
		self.dropout_learning_rate = dropout_learning_rate
		self.dropout_learning_rate_change_stack.append((self.epoch_index, self.dropout_learning_rate))

	def build_functions(self):
		super(DynamicRecurrentNetwork, self).build_functions()

		#
		#
		#
		#
		#

		#
		#
		#
		#
		#

		# Create update expressions for training, i.e., how to modify the parameters at each training step.
		# Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_objective = self.get_objectives(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		if self.total_norm_constraint > 0:
			adaptable_grads = theano.tensor.grad(dropout_loss, adaptable_params)
			scaled_adaptable_grads = updates.total_norm_constraint(adaptable_grads, self.total_norm_constraint)
			adaptable_params_updates = self._update_function(scaled_adaptable_grads, adaptable_params,
			                                                 self._dropout_learning_rate_variable)
		else:
			adaptable_params_updates = self._update_function(dropout_loss, adaptable_params,
			                                                 self._dropout_learning_rate_variable)

		'''
		if self.max_norm_constraint > 0:
			for param in self.get_network_params(adaptable=True):
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

				adaptable_params_updates[param] = updates.norm_constraint(adaptable_params_updates[param],
				                                                          self.max_norm_constraint,
				                                                          norm_axes=sum_over)
		'''

		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable,
			        self._dropout_learning_rate_variable],
			outputs=[dropout_objective, dropout_accuracy],
			updates=adaptable_params_updates
		)

		#
		#
		#
		#
		#

		if False:
			train_loss = dropout_loss
			train_objective = dropout_objective
			train_accuracy = dropout_accuracy

			# Create update expressions for training, i.e., how to modify the parameters at each training step.
			# Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
			trainable_params = self.get_network_params(trainable=True)
			if self.total_norm_constraint > 0:
				trainable_grads = theano.tensor.grad(train_loss, trainable_params)
				scaled_trainable_grads = updates.total_norm_constraint(trainable_grads, self.total_norm_constraint)
				trainable_params_updates = self._update_function(scaled_trainable_grads, trainable_params,
				                                                 self._learning_rate_variable)
			else:
				trainable_params_updates = self._update_function(train_loss, trainable_params,
				                                                 self._learning_rate_variable)

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

		#
		#
		#
		#
		#

		# Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
		train_loss = self.get_loss(self._output_variable)
		train_objective = self.get_objectives(self._output_variable)
		train_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy")

		# Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
		test_loss = self.get_loss(self._output_variable, deterministic=True)
		test_objective = self.get_objectives(self._output_variable, deterministic=True)
		test_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy",
		                                    deterministic=True)

		from lasagne.experiments.debugger import debug_rademacher
		self._debug_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable],
			outputs=[] \
			        # + trainable_grads + adaptable_grads \
			        + debug_rademacher(self, self._output_variable, deterministic=True) \
			        + [train_loss, train_objective, train_accuracy] \
			        + [test_loss, test_objective, test_accuracy] \
			        + [dropout_loss, dropout_objective, dropout_accuracy],
			on_unused_input='ignore'
		)

	#
	#
	#
	#
	#

	def train_minibatch(self, minibatch_x, minibatch_y, minibatch_m):
		#
		#
		#
		#
		#
		'''
		debug_output = self._debug_function(minibatch_x, minibatch_y, minibatch_m)
		format_output = [];
		for model_output in debug_output[:-15]:
			format_output.append((model_output.shape, numpy.max(model_output), numpy.min(model_output)))
		for model_output in debug_output[-15:]:
			format_output.append(model_output)
		print("checkpoing 1:", len(debug_output), format_output)
		'''
		#
		#
		#
		#
		#

		minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy = super(
			DynamicRecurrentNetwork, self).train_minibatch(minibatch_x, minibatch_y, minibatch_m)

		#
		#
		#
		#
		#
		'''
		debug_output = self._debug_function(minibatch_x, minibatch_y, minibatch_m)
		format_output = [];
		for model_output in debug_output[:-15]:
			format_output.append((model_output.shape, numpy.max(model_output), numpy.min(model_output)))
		for model_output in debug_output[-15:]:
			format_output.append(model_output)
		print("checkpoing 2:", len(debug_output), format_output)
		'''
		#
		#
		#
		#
		#

		'''
		dropout_learning_rate = self.dropout_learning_rate
		if self.dropout_learning_rate_decay is not None:
			if self.dropout_learning_rate_decay[0] == "epoch":
				dropout_learning_rate = decay_learning_rate(self.dropout_learning_rate, self.epoch_index,
				                                            self.dropout_learning_rate_decay)
			elif self.dropout_learning_rate_decay[0] == "iteration":
				dropout_learning_rate = decay_learning_rate(self.dropout_learning_rate, self.minibatch_index,
				                                            self.dropout_learning_rate_decay)
		'''

		minibatch_running_time_temp = timeit.default_timer()
		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			dropout_learning_rate = adjust_parameter_according_to_policy(self.dropout_learning_rate, self.epoch_index)
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y, minibatch_m,
			                                                              dropout_learning_rate)
		minibatch_running_time_temp = timeit.default_timer() - minibatch_running_time_temp

		#
		#
		#
		#
		#
		'''
		debug_output = self._debug_function(minibatch_x, minibatch_y, minibatch_m)
		format_output = [];
		for model_output in debug_output[:-15]:
			format_output.append((model_output.shape, numpy.max(model_output), numpy.min(model_output)))
		for model_output in debug_output[-15:]:
			format_output.append(model_output)
		print("checkpoing 3:", len(debug_output), format_output)
		'''
		#
		#
		#
		#
		#

		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time + minibatch_running_time_temp, minibatch_average_train_objective, minibatch_average_train_accuracy