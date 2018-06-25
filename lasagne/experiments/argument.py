import datetime
import logging
import os

from lasagne import Xregularization, Xpolicy
from lasagne import objectives, updates
from . import debug, data

logger = logging.getLogger(__name__)

__all__ = [
	"parse_parameter_policy",
	#
	"layer_deliminator",
	"param_deliminator",
	#
	"add_discriminative_options",
	"validate_discriminative_options",
	#
	"add_resume_options",
	"validate_resume_options",
	#
	"add_adaptive_options",
	"validate_adaptive_options",
]

layer_deliminator = "*"
# layer_deliminator = ","
param_deliminator = ","
# param_deliminator = "*"
specs_deliminator = ":"


def parse_parameter_policy(policy_string):
	policy_tokens = policy_string.split(param_deliminator)

	policy_tokens[0] = float(policy_tokens[0])
	# assert policy_tokens[0] >= 0
	if len(policy_tokens) == 1:
		policy_tokens.append(Xpolicy.constant)
		return policy_tokens

	policy_tokens[1] = getattr(Xpolicy, policy_tokens[1])
	if policy_tokens[1] is Xpolicy.constant:
		assert len(policy_tokens) == 2
		return policy_tokens

	if policy_tokens[1] is Xpolicy.piecewise_constant:
		assert len(policy_tokens) == 4

		policy_tokens[2] = [float(boundary_token) for boundary_token in policy_tokens[2].split("-")]
		previous_boundary = 0
		for next_boundary in policy_tokens[2]:
			assert next_boundary > previous_boundary
			previous_boundary = next_boundary
		policy_tokens[3] = [float(value_token) for value_token in policy_tokens[3].split("-")]
		assert len(policy_tokens[2]) == len(policy_tokens[3])
		return policy_tokens

	assert policy_tokens[1] is Xpolicy.inverse_time_decay \
	       or policy_tokens[1] is Xpolicy.natural_exp_decay \
	       or policy_tokens[1] is Xpolicy.exponential_decay

	for x in range(2, 4):
		policy_tokens[x] = float(policy_tokens[x])
		assert policy_tokens[x] > 0

	if len(policy_tokens) == 4:
		policy_tokens.append(0)
	elif len(policy_tokens) == 5:
		policy_tokens[4] = float(policy_tokens[4])
		assert policy_tokens[4] > 0
	else:
		logger.error("unrecognized parameter decay policy %s..." % (policy_tokens))

	return policy_tokens


def add_generic_options(model_parser):
	# model_parser = argparse.ArgumentParser(description="generic neural network arguments", add_help=True)

	# generic argument set 1
	model_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output directory [None]")

	# generic argument set 2
	model_parser.add_argument("--objective", dest="objective", action='store', default="categorical_crossentropy",
	                          help="objective function [categorical_crossentropy] defined in objectives.py")
	model_parser.add_argument("--update", dest="update", action='store', default="nesterov_momentum",
	                          help="update function [nesterov_momentum] defined updates.py")
	model_parser.add_argument("--regularizer", dest='regularizer', action='append', default=[],
	                          help="regularizer function [None] defined in regularization.py")
	# "'l2:0.1'=l2-regularizer with lambda 0.1 applied over all layers, " +
	# "'l1:0.1;0.2;0.3'=l1-regularizer with lambda 0.1, 0.2, 0.3 applied over three layers"

	# generic argument set 3
	model_parser.add_argument("--minibatch_size", dest="minibatch_size", type=int, action='store', default=-1,
	                          help="mini-batch size [-1]")
	model_parser.add_argument("--number_of_epochs", dest="number_of_epochs", type=int, action='store', default=-1,
	                          help="number of epochs [-1]")
	# generic_parser.add_argument("--snapshot_interval", dest="snapshot_interval", type=int, action='store', default=0,
	# help="snapshot interval in number of epochs [0 - no snapshot]")

	# generic argument set 4
	'''
	generic_parser.add_argument("--learning_rate", dest="learning_rate", type=float, action='store', default=1e-2,
	                            help="learning rate [1e-2]")
	generic_parser.add_argument("--learning_rate_decay", dest="learning_rate_decay", action='store', default=None,
	                            help="learning rate decay [None], example, 'iteration,inverse_t,0.2,0.1', 'epoch,exponential,1.7,0.1', 'epoch,step,0.2,100'")
	'''
	model_parser.add_argument("--learning_rate", dest="learning_rate", action='store', default="1e-2",
	                          help="learning policy [1e-2,constant]")

	model_parser.add_argument("--gradient_max_global_l2_norm", dest="gradient_max_global_l2_norm", type=float,
	                          action='store', default=0, help="max global l2 norm of gradient constraint [0 - None]")
	model_parser.add_argument("--parameter_max_local_l2_norm", dest="parameter_max_local_l2_norm", type=float,
	                          action='store', default=0, help="max local l2 norm of parameter constraint [0 - None]")

	# generic argument set 5
	model_parser.add_argument("--data_pipe", dest='data_pipe', action='store', default="load_features_labels",
	                          help="dataset loading/preprocessing pipeline [None]")

	model_parser.add_argument("--snapshot", dest='snapshot', action='append', default=[],
	                          help="snapshot function [None]")
	model_parser.add_argument("--debug", dest='debug', action='append', default=[], help="debug function [None]")

	'''
	subparsers = generic_parser.add_subparsers(dest="subparser_name")
	resume_parser = subparsers.add_parser("resume", parents=[generic_parser], help='resume training')
	resume_parser = add_resume_options(resume_parser)

	start_parser = subparsers.add_parser('start', parents=[generic_parser], help='start training')
	'''

	return model_parser


def validate_generic_options(arguments):
	# generic argument set 4
	'''
	assert arguments.learning_rate > 0
	if arguments.learning_rate_decay is not None:
		learning_rate_decay_tokens = arguments.learning_rate_decay.split(",")
		assert len(learning_rate_decay_tokens) == 4
		assert learning_rate_decay_tokens[0] in ["iteration", "epoch"]
		assert learning_rate_decay_tokens[1] in ["inverse_t", "exponential", "step"]
		learning_rate_decay_tokens[2] = float(learning_rate_decay_tokens[2])
		learning_rate_decay_tokens[3] = float(learning_rate_decay_tokens[3])
		arguments.learning_rate_decay = learning_rate_decay_tokens
	'''

	# if arguments.subparser_name == "resume":
	#	arguments = validate_resume_arguments(arguments)

	arguments.learning_rate = parse_parameter_policy(arguments.learning_rate)
	assert arguments.parameter_max_local_l2_norm >= 0
	assert arguments.gradient_max_global_l2_norm >= 0

	# generic argument set data pipe
	data_pipe = []
	for data_function in arguments.data_pipe.split(param_deliminator):
		data_function = getattr(data, data_function)
		data_pipe.append(data_function)
	arguments.data_pipe = data_pipe

	# generic argument set snapshots
	snapshots = {}
	for snapshot_interval_mapping in arguments.snapshot:
		fields = snapshot_interval_mapping.split(specs_deliminator)
		snapshot_function = getattr(debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized snapshot function setting %s..." % (snapshot_interval_mapping))
		snapshots[snapshot_function] = interval
	arguments.snapshot = snapshots

	debugs = {}
	for debug_interval_mapping in arguments.debug:
		fields = debug_interval_mapping.split(specs_deliminator)
		debug_function = getattr(debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized debug function setting %s..." % (debug_interval_mapping))
		debugs[debug_function] = interval
	arguments.debug = debugs

	# generic argument set 3
	assert arguments.minibatch_size > 0
	assert arguments.number_of_epochs > 0
	# assert arguments.snapshot_interval >= 0

	# generic argument set 2
	arguments.objective = getattr(objectives, arguments.objective)
	arguments.update = getattr(updates, arguments.update)

	regularizers = {}
	for regularizer_weight_mapping in arguments.regularizer:
		fields = regularizer_weight_mapping.split(specs_deliminator)
		# regularizer_function = getattr(regularization, fields[0])
		regularizer_function = getattr(Xregularization, fields[0])
		if len(fields) == 1:
			regularizers[regularizer_function] = [Xpolicy.constant, 1.0]
		elif len(fields) == 2:
			regularizers[regularizer_function] = parse_parameter_policy(fields[1])
			'''
			tokens = fields[1].split(layer_deliminator)
			if len(tokens) == 1:
				weight = float(tokens[0])
			else:
				weight = [float(token) for token in tokens]
			regularizers[regularizer_function] = weight
			'''
		else:
			logger.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping))
	arguments.regularizer = regularizers

	# generic argument set 1
	# self.input_directory = arguments.input_directory
	assert os.path.exists(arguments.input_directory)

	output_directory = arguments.output_directory
	assert (output_directory is not None)
	if not os.path.exists(output_directory):
		os.mkdir(os.path.abspath(output_directory))
	# adjusting output directory
	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S-%f") + ""
	# suffix += "-%s" % ("mlp")
	output_directory = os.path.join(output_directory, suffix)
	assert not os.path.exists(output_directory)
	# os.mkdir(os.path.abspath(output_directory))
	arguments.output_directory = output_directory

	return arguments


def add_discriminative_options(model_parser):
	model_parser = add_generic_options(model_parser)

	# model argument set
	model_parser.add_argument("--validation_data", dest="validation_data", type=int, action='store', default=0,
	                          help="validation data [0 - no validation data used], -1 - load validate.(feature|label).npy for validation]")
	#model_parser.add_argument("--validation_interval", dest="validation_interval", type=int, action='store',
	                          #default=1000, help="validation interval in number of mini-batches [1000]")

	return model_parser


def validate_discriminative_options(arguments):
	arguments = validate_generic_options(arguments)

	# model argument set
	assert (arguments.validation_data >= -1)
	#assert (arguments.validation_interval > 0)

	return arguments


def add_resume_options(model_parser):
	# from . import add_discriminative_options

	# model_parser = add_discriminative_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def validate_resume_options(arguments):
	# from . import validate_discriminative_options

	# arguments = validate_discriminative_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments


def add_adaptive_options(model_parser):
	# from . import add_discriminative_options
	# model_parser = add_discriminative_options()
	# model_parser.description = "adaptive multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--adaptable_learning_rate", dest="adaptable_learning_rate", action='store',
	                          default=None, help="adaptable learning rate [None - learning_rate]")
	model_parser.add_argument("--adaptable_training_mode", dest="adaptable_training_mode",
	                          action='store', default="train_adaptables_networkwise",
	                          help="train adaptables mode [train_adaptables_networkwise]")
	# model_parser.add_argument("--adaptable_update_interval", dest="adaptable_update_interval", type=int,
	# action='store', default=1, help="adatable update interval [1]")

	return model_parser


def validate_adaptive_options(arguments):
	# from . import validate_discriminative_options
	# arguments = validate_discriminative_options(arguments)

	# model argument set 1
	from . import parse_parameter_policy
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	if arguments.adaptable_learning_rate is None:
		arguments.adaptable_learning_rate = arguments.learning_rate
	else:
		arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)

	assert arguments.adaptable_training_mode in {"train_adaptables_networkwise", "train_adaptables_layerwise",
	                                             "train_adaptables_layerwise_in_turn"}

	# assert (arguments.adaptable_update_interval >= 0)

	return arguments


def discriminative_adaptive_resume_parser():
	from . import add_adaptive_options

	model_parser = add_adaptive_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def discriminative_adaptive_resume_validator(arguments):
	from . import validate_adaptive_options

	arguments = validate_adaptive_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments
