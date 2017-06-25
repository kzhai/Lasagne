import cPickle
import logging
import numpy
import os
import sys
import timeit

from .. import networks
from .. import nonlinearities
from .base import load_data, load_data_to_train_validate

__all__ = [
    "train_mlp",
]

def construct_mlp_parser():
    from .base import construct_generic_parser
    model_parser = construct_generic_parser();

    # model argument set 1
    model_parser.add_argument("--layer_dimensions", dest="layer_dimensions", action='store', default=None,
                              help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    model_parser.add_argument("--layer_nonlinearities", dest="layer_nonlinearities", action='store', default=None,
                              help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");

    # model argument set 2
    model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store', default="1.0",
                              help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store', default="bernoulli",
                              help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");

    # model argument set
    model_parser.add_argument("--validation_interval", dest="validation_interval", type=int, action='store', default=1000,
                              help="validation interval in number of mini-batches [1000]");

    '''
    model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
                              help="pretrained model file [None]");
    model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
                              help="dae regularization lambda [0]")
    model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
                              help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
    '''

    return model_parser;

def validate_mlp_arguments(arguments):
    from .base import validate_generic_arguments
    arguments = validate_generic_arguments(arguments);

    # model argument set 1
    assert arguments.layer_dimensions != None
    arguments.layer_dimensions = [int(dimensionality) for dimensionality in arguments.layer_dimensions.split(",")]
    number_of_layers = len(arguments.layer_dimensions);

    assert arguments.layer_nonlinearities != None
    layer_nonlinearities = arguments.layer_nonlinearities.split(",")
    layer_nonlinearities = [getattr(nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
    assert len(layer_nonlinearities) == number_of_layers;
    arguments.layer_nonlinearities = layer_nonlinearities;

    # model argument set 2
    layer_activation_styles = arguments.layer_activation_styles;
    layer_activation_style_tokens = layer_activation_styles.split(",")
    if len(layer_activation_style_tokens) == 1:
        layer_activation_styles = [layer_activation_styles for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_style_tokens) == number_of_layers:
        layer_activation_styles = layer_activation_style_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    assert len(layer_activation_styles) == number_of_layers;
    assert (layer_activation_style in set(
        ["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli",
         "mixed_beta_bernoulli"]) for layer_activation_style in layer_activation_styles)
    arguments.layer_activation_styles = layer_activation_styles;

    layer_activation_parameters = arguments.layer_activation_parameters;
    layer_activation_parameter_tokens = layer_activation_parameters.split(",")
    if len(layer_activation_parameter_tokens) == 1:
        layer_activation_parameters = [layer_activation_parameters for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_parameter_tokens) == number_of_layers:
        layer_activation_parameters = layer_activation_parameter_tokens
    assert len(layer_activation_parameters) == number_of_layers;

    for layer_index in xrange(number_of_layers):
        if layer_activation_styles[layer_index] == "bernoulli":
            layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
            assert layer_activation_parameters[layer_index] <= 1;
            assert layer_activation_parameters[layer_index] > 0;
        elif layer_activation_styles[layer_index] == "beta_bernoulli" or layer_activation_styles[
            layer_index] == "reciprocal_beta_bernoulli" or layer_activation_styles[
            layer_index] == "reverse_reciprocal_beta_bernoulli" or layer_activation_styles[
            layer_index] == "mixed_beta_bernoulli":
            layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+");
            assert len(layer_activation_parameter_tokens) == 2;
            layer_activation_parameters[layer_index] = (
            float(layer_activation_parameter_tokens[0]), float(layer_activation_parameter_tokens[1]))
            assert layer_activation_parameters[layer_index][0] > 0;
            assert layer_activation_parameters[layer_index][1] > 0;
            if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
                assert layer_activation_parameters[layer_index][0] < 1;
    arguments.layer_activation_parameters = layer_activation_parameters;

    # model argument set
    assert (arguments.validation_interval > 0);

    '''
    dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
    if isinstance(dae_regularizer_lambdas, int):
        dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
    assert len(dae_regularizer_lambdas) == number_of_layers - 1;
    assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
    self.dae_regularizer_lambdas = dae_regularizer_lambdas;

    layer_corruption_levels = arguments.layer_corruption_levels;
    if isinstance(layer_corruption_levels, int):
        layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
    assert len(layer_corruption_levels) == number_of_layers - 1;
    assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
    assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
    self.layer_corruption_levels = layer_corruption_levels;

    pretrained_model_file = arguments.pretrained_model_file;
    pretrained_model = None;
    if pretrained_model_file != None:
        assert os.path.exists(pretrained_model_file)
        pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
    '''

    return arguments

def train_mlp():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """

    arguments, additionals = construct_mlp_parser().parse_known_args();
    #arguments, additionals = model_parser.parse_known_args()

    settings = validate_mlp_arguments(arguments);

    input_directory = settings.input_directory
    output_directory = settings.output_directory
    assert not os.path.exists(output_directory)
    os.mkdir(output_directory);

    logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s');
    validation_data = settings.validation_data
    minibatch_size = settings.minibatch_size

    print "========== ==========", "parameters", "========== =========="
    for key, value in vars(settings).iteritems():
        print "%s=%s" % (key, value);
    print "========== ==========", "additional", "========== =========="
    for addition in additionals:
        print "%s" % (addition);

    logging.info("========== ==========" + "parameters" + "========== ==========")
    for key, value in vars(settings).iteritems():
        logging.info("%s=%s" % (key, value));
    logging.info("========== ==========" + "parameters" + "========== ==========")

    cPickle.dump(settings, open(os.path.join(output_directory, "settings.pkl"), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    #
    #
    #
    #
    #

    test_dataset = load_data(input_directory, dataset="test");
    if validation_data>=0:
        train_dataset_info, validate_dataset_info = load_data_to_train_validate(input_directory, validation_data);
        train_dataset, train_indices = train_dataset_info;
        validate_dataset, validate_indices =validate_dataset_info;
        numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices);
        numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices);
    else:
        train_dataset = load_data(input_directory, dataset="train");
        validate_dataset = load_data(input_directory, dataset="validate");
    train_set_x, train_set_y = train_dataset;
    input_shape = list(train_set_x.shape[1:]);
    input_shape.insert(0, None)
    input_shape = tuple(input_shape)

    #
    #
    #
    #
    #

    network = networks.MultiLayerPerceptron(
        incoming=input_shape,

        layer_dimensions=settings.layer_dimensions,
        layer_nonlinearities=settings.layer_nonlinearities,

        layer_activation_parameters=settings.layer_activation_parameters,
        layer_activation_styles=settings.layer_activation_styles,

        objective_functions=settings.objective,
        update_function=settings.update,
        # pretrained_model=pretrained_model

        learning_rate = settings.learning_rate,
        learning_rate_decay_style=settings.learning_rate_decay_style,
        learning_rate_decay_parameter=settings.learning_rate_decay_parameter,
        validation_interval=settings.validation_interval,
    )

    network.set_regularizers(settings.regularizer);
    #network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
    #network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

    ########################
    # START MODEL TRAINING #
    ########################

    start_train = timeit.default_timer()
    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch_index in range(settings.number_of_epochs):
        network.train(train_dataset, minibatch_size, validate_dataset, test_dataset, output_directory);

        if settings.snapshot_interval>0 and network.epoch_index % settings.snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
            cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

        print "PROGRESS: %f%%" % (100. * epoch_index / settings.number_of_epochs);

    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    end_train = timeit.default_timer()

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print >> sys.stderr, ('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))

def resume_mlp():
    pass

def test_mlp():
    pass

if __name__ == '__main__':
    train_mlp()
