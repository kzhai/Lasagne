import logging
import pickle
import os

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"start_mlpA",
	"resume_mlpA",
]

from . import mlpA_parser, mlpA_validator


def start_mlpA():
	from . import config_model, validate_config
	settings = config_model(mlpA_parser, mlpA_validator)
	settings = validate_config(settings)

	network = networks.AdaptiveFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	mlp = networks.AdaptedMultiLayerPerceptronFromSpecifications(
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


def resume_mlpA():
	from . import config_model, validate_config, discriminative_adaptive_resume_parser, \
		discriminative_adaptive_resume_validator

	settings = config_model(discriminative_adaptive_resume_parser, discriminative_adaptive_resume_validator)
	settings = validate_config(settings)

	network = networks.AdaptiveFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	model = pickle.load(open(settings.model_file, 'rb'))
	mlp = networks.AdaptedMultiLayerPerceptronFromPretrainedModel(
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
		resume_mlpA()
	else:
		start_mlpA()
