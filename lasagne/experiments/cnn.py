import cPickle
import logging
import numpy
import os
import sys
import timeit

from .. import networks
from .. import nonlinearities, objectives
from .base import load_data, load_data_to_train_validate

__all__ = [
    "train_cnn",
]

def construct_cnn_parser():
    from .base import construct_generic_parser
    model_parser = construct_generic_parser();

    model_parser.description = "convolutional neural network argument";

    # model argument set 1
    model_parser.add_argument("--convolution_filters", dest="convolution_filters", action='store', default=None,
                              help="number of convolution filters [None], example, '32,16' represents 32 and 16 filters for convolution layers respectively");
    model_parser.add_argument("--convolution_nonlinearities", dest="convolution_nonlinearities", action='store', default=None,
                              help="activation functions of convolution layers [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");

    # model argument set 2
    model_parser.add_argument("--dense_dimensions", dest="dense_dimensions", action='store', default=None,
                              help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    model_parser.add_argument("--dense_nonlinearities", dest="dense_nonlinearities", action='store', default=None,
                              help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");

    # model argument set 3
    model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store', default="1.0",
                              help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store', default="bernoulli",
                              help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");

    # model argument set 4
    model_parser.add_argument("--convolution_filter_sizes", dest="convolution_filter_sizes", action='store', default="5*5",
                              help="convolution filter sizes [5*5], example, '5*5,6*6' represents 5*5 and 6*6 filter size for convolution layers respectively");
    model_parser.add_argument("--convolution_strides", dest="convolution_strides", action='store', default="1*1",
                              help="convolution strides [1*1], example, '1*1,2*2' represents 1*1 and 2*2 stride size for convolution layers respectively");
    model_parser.add_argument("--convolution_pads", dest="convolution_pads", action='store', default="2",
                              help="convolution pads [2], example, '2,3' represents 2 and 3 pads for convolution layers respectively");

    # model argument set 5
    model_parser.add_argument("--pooling_sizes", dest="pooling_sizes", action='store', default="3*3",
                              help="pooling sizes [3*3], example, '2*2,3*3' represents 2*2 and 3*3 pooling size respectively");
    model_parser.add_argument("--pooling_strides", dest="pooling_strides", action='store', default="2*2",
                              help="pooling strides [2*2], example, '2*2,3*3' represents 2*2 and 3*3 pooling stride respectively");

    #
    #
    #
    #
    #

    '''
    # model argument set
    model_parser.add_argument("--L1_regularizer_lambdas", dest="L1_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
                              help="L1 regularization lambda [0]")
    model_parser.add_argument("--L2_regularizer_lambdas", dest="L2_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
                              help="L2 regularization lambda [0]")
    model_parser.add_argument("--max_norm_regularizer_lambdas", dest="max_norm_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
                              help="max norm regularizer [0 - no max norm regularization, normally set to a value between 3 and 4]")
    '''

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

    return model_parser

'''
class CNNConfiguration(Configuration):
    def __init__(self, arguments):
        super(CNNConfiguration, self).__init__(arguments);

        # model argument set 1
        assert arguments.convolution_filters != None
        convolution_filters = arguments.convolution_filters.split(",")
        self.convolution_filters = [int(convolution_filter) for convolution_filter in convolution_filters]

        assert arguments.convolution_nonlinearities != None
        convolution_nonlinearities = arguments.convolution_nonlinearities.split(",")
        self.convolution_nonlinearities = [getattr(nonlinearities, layer_nonlinearity) for layer_nonlinearity in convolution_nonlinearities]

        assert len(convolution_nonlinearities) == len(convolution_filters)
        number_of_convolution_layers = len(self.convolution_filters);

        # model argument set 2
        assert arguments.dense_dimensions != None
        dense_dimensions = arguments.dense_dimensions.split(",")
        self.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

        assert arguments.dense_nonlinearities != None
        dense_nonlinearities = arguments.dense_nonlinearities.split(",")
        self.dense_nonlinearities = [getattr(nonlinearities, layer_nonlinearity) for layer_nonlinearity in dense_nonlinearities]

        assert len(dense_dimensions) == len(dense_nonlinearities)
        number_of_dense_layers = len(self.dense_dimensions);

        number_of_layers = number_of_convolution_layers + number_of_dense_layers;

        # model argument set 3
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
        self.layer_activation_styles = layer_activation_styles;

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
        self.layer_activation_parameters = layer_activation_parameters;

        #
        #
        #
        #
        #

        convolution_filter_size_tokens = convolution_filter_sizes.split(",")
        if len(convolution_filter_size_tokens) == 1:
            convolution_filter_sizes = [tuple(int(x) for x in convolution_filter_size_tokens.split("*")) for layer_index in xrange(number_of_convolution_layers)]
        elif len(layer_activation_parameter_tokens) == number_of_convolution_layers:
            convolution_filter_sizes = [tuple(int(x) for x in convolution_filter_sizes_token.split("*")) for
                                        convolution_filter_sizes_token in convolution_filter_size_tokens]
        assert len(convolution_filter_sizes) == number_of_convolution_layers;

        # model argument set 4
        convolution_filter_sizes = arguments.convolution_filter_sizes;
        convolution_filter_sizes = [tuple(int(x) for x in token.split("*")) for token in convolution_filter_sizes.split(",")];
        if len(convolution_filter_sizes)==1:
            convolution_filter_sizes *= number_of_convolution_layers;
        assert len(convolution_filter_sizes) == number_of_convolution_layers;
        self.convolution_filter_sizes = convolution_filter_sizes

        convolution_strides = arguments.convolution_strides
        convolution_strides = [tuple(int(x) for x in token.split("*")) for token in convolution_strides.split(",")];
        if len(convolution_strides)==1:
            convolution_strides *= number_of_convolution_layers;
        assert len(convolution_strides)==number_of_convolution_layers;
        self.convolution_strides = convolution_strides

        convolution_pads = arguments.convolution_pads
        convolution_pads = [int(x) for x in convolution_pads.split(",")];
        if len(convolution_pads) == 1:
            convolution_pads *= number_of_convolution_layers;
        assert len(convolution_pads) == number_of_convolution_layers;
        self.convolution_pads = convolution_pads

        # model argument set 5
        pooling_sizes = arguments.pooling_sizes;
        pooling_sizes = [tuple(int(x) for x in token.split("*")) for token in pooling_sizes.split(",")];
        if len(pooling_sizes) == 1:
            pooling_sizes *= number_of_convolution_layers;
        assert len(pooling_sizes) == number_of_convolution_layers;
        self.pooling_sizes=pooling_sizes

        pooling_strides = arguments.pooling_strides;
        pooling_strides = [tuple(int(x) for x in token.split("*")) for token in pooling_strides.split(",")];
        if len(pooling_strides) == 1:
            pooling_strides *= number_of_convolution_layers;
        assert len(pooling_strides) == number_of_convolution_layers;
        self.pooling_strides = pooling_strides;

        #
        #
        #
        #
        #

        # model argument set
        assert (arguments.validation_interval > 0);
        self.validation_interval = arguments.validation_interval;

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

def validate_cnn_arguments(arguments):
    from .base import validate_generic_arguments
    arguments = validate_generic_arguments(arguments);

    # model argument set 1
    assert arguments.convolution_filters != None
    convolution_filters = arguments.convolution_filters.split(",")
    arguments.convolution_filters = [int(convolution_filter) for convolution_filter in convolution_filters]

    assert arguments.convolution_nonlinearities != None
    convolution_nonlinearities = arguments.convolution_nonlinearities.split(",")
    arguments.convolution_nonlinearities = [getattr(nonlinearities, layer_nonlinearity) for layer_nonlinearity in convolution_nonlinearities]

    assert len(convolution_nonlinearities) == len(convolution_filters)
    number_of_convolution_layers = len(arguments.convolution_filters);

    # model argument set 2
    assert arguments.dense_dimensions != None
    dense_dimensions = arguments.dense_dimensions.split(",")
    arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

    assert arguments.dense_nonlinearities != None
    dense_nonlinearities = arguments.dense_nonlinearities.split(",")
    arguments.dense_nonlinearities = [getattr(nonlinearities, layer_nonlinearity) for layer_nonlinearity in dense_nonlinearities]

    assert len(dense_dimensions) == len(dense_nonlinearities)
    number_of_dense_layers = len(arguments.dense_dimensions);

    number_of_layers = number_of_convolution_layers + number_of_dense_layers;

    # model argument set 3
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

    #
    #
    #
    #
    #

    '''
    convolution_filter_size_tokens = convolution_filter_sizes.split(",")
    if len(convolution_filter_size_tokens) == 1:
        convolution_filter_sizes = [tuple(int(x) for x in convolution_filter_size_tokens.split("*")) for layer_index in xrange(number_of_convolution_layers)]
    elif len(layer_activation_parameter_tokens) == number_of_convolution_layers:
        convolution_filter_sizes = [tuple(int(x) for x in convolution_filter_sizes_token.split("*")) for
                                    convolution_filter_sizes_token in convolution_filter_size_tokens]
    assert len(convolution_filter_sizes) == number_of_convolution_layers;
    '''

    # model argument set 4
    convolution_filter_sizes = arguments.convolution_filter_sizes;
    convolution_filter_sizes = [tuple(int(x) for x in token.split("*")) for token in convolution_filter_sizes.split(",")];
    if len(convolution_filter_sizes)==1:
        convolution_filter_sizes *= number_of_convolution_layers;
    assert len(convolution_filter_sizes) == number_of_convolution_layers;
    arguments.convolution_filter_sizes = convolution_filter_sizes

    convolution_strides = arguments.convolution_strides
    convolution_strides = [tuple(int(x) for x in token.split("*")) for token in convolution_strides.split(",")];
    if len(convolution_strides)==1:
        convolution_strides *= number_of_convolution_layers;
    assert len(convolution_strides)==number_of_convolution_layers;
    arguments.convolution_strides = convolution_strides

    convolution_pads = arguments.convolution_pads
    convolution_pads = [int(x) for x in convolution_pads.split(",")];
    if len(convolution_pads) == 1:
        convolution_pads *= number_of_convolution_layers;
    assert len(convolution_pads) == number_of_convolution_layers;
    arguments.convolution_pads = convolution_pads

    # model argument set 5
    pooling_sizes = arguments.pooling_sizes;
    pooling_sizes = [tuple(int(x) for x in token.split("*")) for token in pooling_sizes.split(",")];
    if len(pooling_sizes) == 1:
        pooling_sizes *= number_of_convolution_layers;
    assert len(pooling_sizes) == number_of_convolution_layers;
    arguments.pooling_sizes=pooling_sizes

    pooling_strides = arguments.pooling_strides;
    pooling_strides = [tuple(int(x) for x in token.split("*")) for token in pooling_strides.split(",")];
    if len(pooling_strides) == 1:
        pooling_strides *= number_of_convolution_layers;
    assert len(pooling_strides) == number_of_convolution_layers;
    arguments.pooling_strides = pooling_strides;

    #
    #
    #
    #
    #

    '''
    L1_regularizer_lambdas = arguments.L1_regularizer_lambdas
    if isinstance(L1_regularizer_lambdas, int):
        L1_regularizer_lambdas = [L1_regularizer_lambdas] * number_of_layers
    assert len(L1_regularizer_lambdas) == number_of_layers
    assert (L1_regularizer_lambda >= 0 for L1_regularizer_lambda in L1_regularizer_lambdas)
    self.L1_regularizer_lambdas = L1_regularizer_lambdas;

    L2_regularizer_lambdas = arguments.L2_regularizer_lambdas
    if isinstance(L2_regularizer_lambdas, int):
        L2_regularizer_lambdas = [L2_regularizer_lambdas] * number_of_layers
    assert len(L2_regularizer_lambdas) == number_of_layers;
    assert (L2_regularizer_lambda >= 0 for L2_regularizer_lambda in L2_regularizer_lambdas)
    self.L2_regularizer_lambdas = L2_regularizer_lambdas;

    #assert arguments.max_norm_regularizer_lambdas >= 0;
    max_norm_regularizer_lambdas = arguments.max_norm_regularizer_lambdas;
    if isinstance(max_norm_regularizer_lambdas, int):
        max_norm_regularizer_lambdas = [max_norm_regularizer_lambdas] * number_of_layers
    assert len(max_norm_regularizer_lambdas) == number_of_layers;
    assert (max_norm_regularizer_lambda >= 0 for max_norm_regularizer_lambda in max_norm_regularizer_lambdas)
    self.max_norm_regularizer_lambdas = max_norm_regularizer_lambdas;
    '''

    # model argument set
    assert (arguments.validation_interval > 0);
    arguments.validation_interval = arguments.validation_interval;

    '''
    dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
    if isinstance(dae_regularizer_lambdas, int):
        dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
    assert len(dae_regularizer_lambdas) == number_of_layers - 1;
    assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
    arguments.dae_regularizer_lambdas = dae_regularizer_lambdas;

    layer_corruption_levels = arguments.layer_corruption_levels;
    if isinstance(layer_corruption_levels, int):
        layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
    assert len(layer_corruption_levels) == number_of_layers - 1;
    assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
    assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
    arguments.layer_corruption_levels = layer_corruption_levels;

    pretrained_model_file = arguments.pretrained_model_file;
    pretrained_model = None;
    if pretrained_model_file != None:
        assert os.path.exists(pretrained_model_file)
        pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
    '''

    return arguments

def train_cnn():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """

    arguments, additionals = construct_cnn_parser().parse_known_args()
    settings = validate_cnn_arguments(arguments);

    input_directory = settings.input_directory
    output_directory = settings.output_directory
    assert not os.path.exists(output_directory)
    os.mkdir(output_directory);

    logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s');
    validation_data = settings.validation_data
    minibatch_size = settings.minibatch_size

    print "========== ========== ========== ========== =========="
    for key, value in vars(settings).iteritems():
        print "%s=%s" % (key, value);
    print "========== ========== ========== ========== =========="

    logging.info("========== ========== ========== ========== ==========")
    for key, value in vars(settings).iteritems():
        logging.info("%s=%s" % (key, value));
    logging.info("========== ========== ========== ========== ==========")

    cPickle.dump(settings, open(os.path.join(output_directory, "settings.pkl"), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    #
    #
    #
    #
    #

    test_dataset = load_data(input_directory, dataset="test")

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

    network = networks.ConvolutionalNeuralNetwork(
        incoming=input_shape,

        convolution_filters=settings.convolution_filters,
        convolution_nonlinearities=settings.convolution_nonlinearities,
        # convolution_filter_sizes=None,
        # maxpooling_sizes=None,

        dense_dimensions=settings.dense_dimensions,
        dense_nonlinearities=settings.dense_nonlinearities,

        layer_activation_parameters=settings.layer_activation_parameters,
        layer_activation_styles=settings.layer_activation_styles,

        objective_functions=settings.objective,
        update_function=settings.update,

        learning_rate = settings.learning_rate,
        learning_rate_decay_style=settings.learning_rate_decay_style,
        learning_rate_decay_parameter=settings.learning_rate_decay_parameter,
        validation_interval=settings.validation_interval,

        convolution_filter_sizes=settings.convolution_filter_sizes,
        convolution_strides=settings.convolution_strides,
        convolution_pads=settings.convolution_pads,

        pooling_sizes=settings.pooling_sizes,
        pooling_strides=settings.pooling_strides
    )
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

    '''
    now = datetime.datetime.now();
    snapshot_index = now.strftime("%y%m%d-%H%M%S");
    snapshot_directory = os.path.join(output_directory, snapshot_index);
    assert not os.path.exists(snapshot_directory);
    os.mkdir(snapshot_directory);

    shutil.copy(os.path.join(output_directory, 'model.pkl'), os.path.join(snapshot_directory, 'model.pkl'));
    snapshot_pattern = re.compile(r'^model\-\d+.pkl$');
    for file_name in os.listdir(output_directory):
        if not re.match(snapshot_pattern, file_name):
            continue;
        shutil.move(os.path.join(output_directory, file_name), os.path.join(snapshot_directory, file_name));
    shutil.move(os.path.join(output_directory, 'settings.pkl'), os.path.join(snapshot_directory, 'settings.pkl'));
    '''

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print >> sys.stderr, ('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))

def resume_cnn():
    pass

def test_cnn():
    pass

'''
input_shape = list(data_x.shape[1:]);
    input_shape.insert(0, None)
    input_shape = tuple(input_shape)
    assert numpy.all(list(input_shape[1:]) == list(test_set_x.shape[1:])), (input_shape[1:], test_set_x.shape[1:]);
'''

if __name__ == '__main__':
    train_cnn()