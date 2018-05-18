import logging
import os

from ..experiments import layer_deliminator, param_deliminator
from ..experiments import parse_parameter_policy, add_adaptive_options, validate_adaptive_options

logger = logging.getLogger(__name__)

__all__ = [
	"add_dynamic_options",
	"validate_dynamic_options",
	#
	# "discriminative_adaptive_dynamic_resume_parser",
	# "discriminative_adaptive_dynamic_resume_validator",
	#
	# "discriminative_adaptive_resume_parser",
	# "discriminative_adaptive_resume_validator",
]


def add_dynamic_options(model_parser):
	# from . import add_adaptive_options
	# model_parser = add_adaptive_options()
	# model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--prune_thresholds", dest="prune_thresholds", action='store', default="-0.001",
	                          # default="-1e3,piecewise_constant,100,1e-3",
	                          # help="prune thresholds [-1e3,piecewise_constant,100,1e-3]"
	                          help="prune thresholds [None]"
	                          )
	model_parser.add_argument("--split_thresholds", dest="split_thresholds", action='store', default="1.001",
	                          # default="1e3,piecewise_constant,100,0.999",
	                          # help="split thresholds [1e3,piecewise_constant,100,0.999]"
	                          help="split thresholds [None]"
	                          )

	# model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default=1,
	# type=int, help="prune split interval [1]")
	model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default="10",
	                          help="prune split interval [10]")

	return model_parser


def validate_dynamic_options(arguments):
	# from . import validate_adaptive_options
	# arguments = validate_adaptive_options(arguments)

	# model argument set 1
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	# number_of_layers = sum(layer_activation_type is layers.DynamicDropoutLayer for layer_activation_type in arguments.layer_activation_types)

	if arguments.prune_thresholds is not None:
		prune_thresholds = arguments.prune_thresholds
		prune_thresholds_tokens = prune_thresholds.split(layer_deliminator)
		prune_thresholds = [parse_parameter_policy(prune_thresholds_token) for prune_thresholds_token in
		                    prune_thresholds_tokens]
		# if len(prune_thresholds) == 1:
		# prune_thresholds *= number_of_layers
		# assert len(prune_thresholds) == number_of_layers
		arguments.prune_thresholds = prune_thresholds

	if arguments.split_thresholds is not None:
		split_thresholds = arguments.split_thresholds
		split_thresholds_tokens = split_thresholds.split(layer_deliminator)
		split_thresholds = [parse_parameter_policy(split_thresholds_token) for split_thresholds_token in
		                    split_thresholds_tokens]
		# if len(split_thresholds) == 1:
		# split_thresholds *= number_of_layers
		# assert len(split_thresholds) == number_of_layers
		arguments.split_thresholds = split_thresholds

	prune_split_interval = arguments.prune_split_interval
	prune_split_interval_tokens = [int(prune_split_interval_token) for prune_split_interval_token in
	                               prune_split_interval.split(param_deliminator)]
	if len(prune_split_interval_tokens) == 1:
		prune_split_interval_tokens.insert(0, 0)
	assert prune_split_interval_tokens[1] >= 0
	arguments.prune_split_interval = prune_split_interval_tokens

	return arguments


def discriminative_adaptive_dynamic_resume_parser():
	from . import add_dynamic_options

	model_parser = add_dynamic_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def discriminative_adaptive_dynamic_resume_validator(arguments):
	from . import validate_dynamic_options

	arguments = validate_dynamic_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments


def discriminative_adaptive_resume_parser():
	model_parser = add_adaptive_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def discriminative_adaptive_resume_validator(arguments):
	arguments = validate_adaptive_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments
