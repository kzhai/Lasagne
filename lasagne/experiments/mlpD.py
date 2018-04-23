import logging
import pickle

from . import layer_deliminator, param_deliminator
from .. import networks, layers

logger = logging.getLogger(__name__)

__all__ = [
	"start_mlpD",
	"resume_mlpD",
]

from . import mlpD_parser, mlpD_validator


def start_mlpD():
	from . import config_model, validate_config
	settings = config_model(mlpD_parser, mlpD_validator)
	settings = validate_config(settings)

	network = networks.DynamicFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		#
		prune_threshold_policies=settings.prune_threshold_policies,
		split_threshold_policies=settings.split_threshold_policies,
		prune_split_interval=settings.prune_split_interval,
		#
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	mlp = networks.DynamicMultiLayerPerceptronFromSpecifications(
		network._input_layer,
		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,
		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles
	)

	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	from . import start_training
	start_training(network, settings)


def resume_mlpD():
	from . import config_model, validate_config, discriminative_adaptive_dynamic_resume_parser, \
		discriminative_adaptive_dynamic_resume_validator

	settings = config_model(discriminative_adaptive_dynamic_resume_parser,
	                        discriminative_adaptive_dynamic_resume_validator)
	settings = validate_config(settings)

	network = networks.DynamicFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		#
		prune_threshold_policies=settings.prune_thresholds,
		split_threshold_policies=settings.split_thresholds,
		prune_split_interval=settings.prune_split_interval,
		#
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	model = pickle.load(open(settings.model_file, 'rb'))
	mlp = networks.DynamicMultiLayerPerceptronFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	from . import resume_training
	resume_training(network, settings)


if __name__ == '__main__':
	import argparse

	model_selector = argparse.ArgumentParser(description="mode selector")
	model_selector.add_argument("--resume", dest="resume", action='store_true', default=False,
	                            help="resume [None]")

	arguments, additionals = model_selector.parse_known_args()

	if arguments.resume:
		resume_mlpD()
	else:
		start_mlpD()
