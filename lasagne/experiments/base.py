import logging
import os
import pickle
import random
import sys
import timeit

import numpy

from . import debug, data

logger = logging.getLogger(__name__)

__all__ = [
	"validate_config",
	#
	"start_training",
	"resume_training",
	#
	# "split_data",
	# "load_mnist",
]

'''
def split_indices(data_size, number_of_validate_data=0):
	assert number_of_validate_data >= 0 and number_of_validate_data < data_size
	indices = numpy.random.permutation(data_size)
	train_indices = indices[number_of_validate_data:]
	validate_indices = indices[:number_of_validate_data]
	return train_indices, validate_indices

def split_data(input_directory, dataset, split_indices):
	data_x, data_y = dataset
	train_indices, validate_indices = split_indices

	train_set_x = data_x[train_indices]
	train_set_y = data_y[train_indices]
	dataset = (train_set_x, train_set_y)
	# numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
	logger.info("successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

	if len(validate_indices) > 0:
		validate_set_x = data_x[validate_indices]
		validate_set_y = data_y[validate_indices]
		validate_dataset = (validate_set_x, validate_set_y)
		# numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)
		logger.info("successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))
	else:
		validate_dataset = None

	return (dataset, train_indices), (validate_dataset, validate_indices)
'''


def split_train_data_to_cross_validate(input_directory, number_of_folds=5, output_directory=None):
	data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
	data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
	assert len(data_x) == len(data_y)
	number_of_data = len(data_y)

	assert number_of_folds >= 0 and number_of_folds < len(data_y)
	split_indices = list(range(0, number_of_data, number_of_data / number_of_folds))
	if len(split_indices) == number_of_folds + 1:
		split_indices[-1] = number_of_data
	elif len(split_indices) == number_of_folds:
		split_indices.append(number_of_data)
	else:
		logger.error("something went wrong...")
		sys.exit()
	assert len(split_indices) == number_of_folds + 1
	indices = list(range(len(data_y)))
	random.shuffle(indices)

	if output_directory is None:
		output_directory = input_directory

	fold_index = 0
	for start_index, end_index in zip(split_indices[:-1], split_indices[1:]):
		fold_output_directory = os.path.join(output_directory, "folds.%d.index.%d" % (number_of_folds, fold_index))
		if not os.path.exists(fold_output_directory):
			os.mkdir(fold_output_directory)

		train_indices = indices[:start_index] + indices[end_index:]
		test_indices = indices[start_index:end_index]
		assert len(train_indices) > 0
		assert len(test_indices) > 0

		train_set_x = data_x[train_indices]
		train_set_y = data_y[train_indices]
		numpy.save(os.path.join(fold_output_directory, "train.feature.npy"), train_set_x)
		numpy.save(os.path.join(fold_output_directory, "train.label.npy"), train_set_y)
		numpy.save(os.path.join(fold_output_directory, "train.index.npy"), train_indices)

		test_set_x = data_x[test_indices]
		test_set_y = data_y[test_indices]
		numpy.save(os.path.join(fold_output_directory, "test.feature.npy"), test_set_x)
		numpy.save(os.path.join(fold_output_directory, "test.label.npy"), test_set_y)
		numpy.save(os.path.join(fold_output_directory, "test.index.npy"), test_indices)

		logger.info(
			"successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
		logger.info("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

		print("successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
		print("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

		fold_index += 1

	return


def validate_config(settings):
	# test_dataset = load_features_labels(settings.input_directory, dataset="test")
	dataset_loading_functions = settings.data_pipe
	input_directory = settings.input_directory
	test_dataset = dataset_loading_functions[0](input_directory, dataset="test")
	for dataset_loading_function in dataset_loading_functions[1:]:
		test_dataset = dataset_loading_function(test_dataset)

	test_set_x, test_set_y = test_dataset
	input_shape = list(test_set_x.shape[1:])
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)
	settings.input_shape = input_shape

	return settings


def _load_datasets_to_start(input_directory, output_directory, number_of_validate_data,
                            dataset_loading_functions=[data.load_features_labels]):
	test_dataset = dataset_loading_functions[0](input_directory, dataset="test")
	for dataset_loading_function in dataset_loading_functions[1:]:
		test_dataset = dataset_loading_function(test_dataset)

	if number_of_validate_data >= 0:
		train_dataset_temp = dataset_loading_functions[0](input_directory, dataset="train")
		for dataset_loading_function in dataset_loading_functions[1:]:
			train_dataset_temp = dataset_loading_function(train_dataset_temp)

		total_data_x, total_data_y = train_dataset_temp

		assert number_of_validate_data >= 0 and number_of_validate_data < len(total_data_y)
		indices = numpy.random.permutation(len(total_data_y))
		train_indices = indices[number_of_validate_data:]
		validate_indices = indices[:number_of_validate_data]

		numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
		numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)

		train_set_x = total_data_x[train_indices]
		train_set_y = total_data_y[train_indices]
		train_dataset = (train_set_x, train_set_y)
		logger.info("successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

		validate_set_x = total_data_x[validate_indices]
		validate_set_y = total_data_y[validate_indices]
		validate_dataset = (validate_set_x, validate_set_y)
		logger.info("successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))
	else:
		train_dataset = dataset_loading_functions[0](input_directory, dataset="train")
		for dataset_loading_function in dataset_loading_functions[1:]:
			train_dataset = dataset_loading_function(train_dataset)

		validate_dataset = dataset_loading_functions[0](input_directory, dataset="validate")
		for dataset_loading_function in dataset_loading_functions[1:]:
			validate_dataset = dataset_loading_function(validate_dataset)

	return train_dataset, validate_dataset, test_dataset


def _load_datasets_to_resume(input_directory, model_directory, output_directory,
                             dataset_loading_functions=[data.load_features_labels]):
	test_dataset = dataset_loading_functions[0](input_directory, dataset="test")
	for dataset_loading_function in dataset_loading_functions[1:]:
		test_dataset = dataset_loading_function(test_dataset)

	train_dataset_temp = dataset_loading_functions[0](input_directory, dataset="train")
	for dataset_loading_function in dataset_loading_functions[1:]:
		train_dataset_temp = dataset_loading_function(train_dataset_temp)

	total_data_x, total_data_y = train_dataset_temp

	train_indices = numpy.load(os.path.join(model_directory, "train.index.npy"))
	validate_indices = numpy.load(os.path.join(model_directory, "validate.index.npy"))

	numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
	numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)

	train_set_x = total_data_x[train_indices]
	train_set_y = total_data_y[train_indices]
	train_dataset = (train_set_x, train_set_y)
	logger.info("successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

	validate_set_x = total_data_x[validate_indices]
	validate_set_y = total_data_y[validate_indices]
	validate_dataset = (validate_set_x, validate_set_y)
	logger.info("successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))

	return train_dataset, validate_dataset, test_dataset


def start_training(network, settings):
	input_directory = settings.input_directory
	output_directory = settings.output_directory
	assert not os.path.exists(output_directory)
	os.mkdir(output_directory)

	datasets = _load_datasets_to_start(input_directory, output_directory, settings.validation_data,
	                                   settings.data_pipe)

	_train_model(network, settings, datasets)


def resume_training(network, settings):
	input_directory = settings.input_directory
	output_directory = settings.output_directory
	assert not os.path.exists(output_directory)
	os.mkdir(output_directory)

	model_directory = settings.model_directory
	assert os.path.exists(model_directory)

	datasets = _load_datasets_to_resume(input_directory, model_directory, output_directory,
	                                    settings.data_pipe)

	_train_model(network, settings, datasets)


def _train_model(network, settings, datasets):
	output_directory = settings.output_directory
	assert os.path.exists(output_directory)

	logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG,
	                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

	train_dataset, validate_dataset, test_dataset = datasets

	if debug.subsample_dataset in settings.debug:
		train_dataset, validate_dataset, test_dataset = debug.subsample_dataset(train_dataset, validate_dataset,
		                                                                        test_dataset)
	'''
	if dataset_preprocessing_functions is not None:
		train_dataset = dataset_preprocessing_functions(train_dataset)
		validate_dataset = dataset_preprocessing_functions(validate_dataset)
		test_dataset = dataset_preprocessing_functions(test_dataset)
	'''

	print("========== ==========", "parameters", "========== ==========")
	for key, value in list(vars(settings).items()):
		print("%s=%s" % (key, value))
	print("========== ==========", "parameters", "========== ==========")

	logger.info("========== ==========" + "parameters" + "========== ==========")
	for key, value in list(vars(settings).items()):
		logger.info("%s=%s" % (key, value))
	logger.info("========== ==========" + "parameters" + "========== ==========")

	model_file_path = os.path.join(output_directory, 'model-0.pkl')
	pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	########################
	# START MODEL TRAINING #
	########################

	for snapshot_function in settings.snapshot:
		snapshot_function(network, settings)

	if debug.display_architecture in settings.debug:
		debug.display_architecture(network)

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	for epoch_index in range(settings.number_of_epochs):
		if debug.debug_rademacher_p_2_q_2 in settings.debug:
			debug.debug_rademacher_p_2_q_2(network, train_dataset)
			debug.debug_rademacher_p_2_q_2(network, test_dataset)
		# if debugger.debug_rademacher_p_1_q_inf in settings.debug:
		# debugger.debug_rademacher_p_1_q_inf(network, train_dataset)
		# debugger.debug_rademacher_p_1_q_inf(network, test_dataset)
		if debug.debug_rademacher_p_inf_q_1 in settings.debug:
			debug.debug_rademacher_p_inf_q_1(network, train_dataset)
			debug.debug_rademacher_p_inf_q_1(network, test_dataset)

		if debug.debug_function_output in settings.debug:
			function_outputs_file = os.path.join(settings.output_directory,
			                                     "function_outputs_train.epoch_%d.npy" % (network.epoch_index))
			debug.debug_function_output(network, train_dataset, minibatch_size=settings.minibatch_size,
			                            output_file=function_outputs_file)
		if debug.debug_l2_norm in settings.debug:
			debug.debug_l2_norm(network, settings)
		# debugger.debug_function_output(network, test_dataset)

		network.train(train_dataset, settings.minibatch_size)

		if validate_dataset is not None and len(validate_dataset[1]) > 0:
			output_file = None
			if output_directory is not None:
				output_file = os.path.join(output_directory, 'model.pkl')
			network.validate(validate_dataset, output_file)

		if test_dataset is not None:
			# if output_directory != None:
			# output_file = os.path.join(output_directory, 'model-%d.pkl' % self.epoch_index)
			# cPickle.dump(self, open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
			network.test(test_dataset)

		network.epoch_index += 1

		# if settings.snapshot_interval > 0 and network.epoch_index % settings.snapshot_interval == 0:
		# model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
		# pickle.dump(network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		for snapshot_function in settings.snapshot:
			if network.epoch_index % settings.snapshot[snapshot_function] == 0:
				snapshot_function(network, settings)

		print("PROGRESS: %f%%" % (100. * (epoch_index + 1) / settings.number_of_epochs))

	if debug.debug_function_output in settings.debug:
		function_outputs_file = os.path.join(settings.output_directory,
		                                     "function_outputs_train.epoch_%d.npy" % (network.epoch_index))
		debug.debug_function_output(network, train_dataset, minibatch_size=settings.minibatch_size,
		                            output_file=function_outputs_file)
	# debugger.debug_function_output(network, test_dataset)

	model_file_path = os.path.join(output_directory, 'model.pkl')
	pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	end_train = timeit.default_timer()

	print("Optimization complete...")
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))


if __name__ == '__main__':
	input_directory = sys.argv[1]
	number_of_folds = int(sys.argv[2])
	split_train_data_to_cross_validate(input_directory, number_of_folds)
