# Lasagne/lasagne/experiments

This is an extension of [Lasagne](https://github.com/Lasagne/Lasagne) with a more user-friendly API.

The package includes a pre-processed copy of [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for experimental purpose.
Each image is reshaped to a 784 dimension float vector, and all pixels are unitized to a float value between 0 and 1.  

The package also features an implementation of [adaptive dropout learning](https://www.microsoft.com/en-us/research/publication/adaptive-dropout-rademacher-complexity-regularization/) as described in 

> Ke Zhai and Huan Wang, [Adaptive Dropout with Rademacher Complexity Regularization](https://openreview.net/pdf?id=S1uxsye0Z), International Conference on Learning Representations (ICLR-2018), Apr 2018.

Please send any bugs or problems to Ke Zhai, _fullname_@microsoft.com.

## Prepare the Environment

Assume the Lasagne package is downloaded to directory `$PROJECT_SPACE/`, i.e., 

	$PROJECT_SPACE/Lasagne/

To prepare the example MNIST dataset,

	cd $PROJECT_SPACE/Lasagne/
	tar zxvf data/mnist_784.tar.gz
	
You should see the data directory called `mnist_784/` containing the `[test|train].[feature|label].npy` files.

## Start Multi-Layer Perceptron

To start a multi-layer perceptron with one hidden layer of 100 ReLU neurons,
	
	python -um lasagne.experiments.mlp start-mlp \
		--input_directory=./mnist_784/ \
		--output_directory=./mnist_784/ \
		--minibatch_size=100 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9,5 \
		--dense_dimensions=100*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_parameters=0.8*0.5 \
		--debug=subsample_dataset \
		--debug=display_architecture

The `dense_dimensions` option specifies the layer dimensionality of the network.
The `dense_nonlinearities` option specifies the corresponding layer activation function, defined in [this file](../nonlinearities.py).
The `debug` option specifies the debug function defined in [this file](../experiments/debug.py).

To launch the MLP with adaptive dropout using Rademacher complexity regularization, 

	python -um lasagne.experiments.mlp start-mlpA \
		--input_directory=./mnist_784/ \
		--output_directory=./mnist_784/ \
		--minibatch_size=100 \
		--number_of_epochs=100 \
		--learning_rate=0.01,exponential_decay,20,0.5 \
		--dense_dimensions=1024*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=AdaptiveDropoutLayer \
		--layer_activation_parameters=0.8*0.5 \
		--regularizer=rademacher:0.01 \
		--snapshot=snapshot_dropout

The `layer_activation_types` option specifies the dropout style you want to use after each layer, defined in [this file](../layers/Xnoise.py).
The `regularizer` option specifies the regularizer function you want to impose to the loss, defined in [this file](../Xregularizer.py).
You can stack arbitrary number of regularizers when you train a model, i.e., specify multiple `--regularizer` arguments, with `:` specifying the corresponding weight.
The `snapshot` option specifies the snapshot function to store the intermediate states during training, defined in [this file](../experiments/debug.py).

You may use arbitrary combination of `layer_activation_types`, for example, standard BernoulliDropoutLayer for the first layer, and AdaptiveDropoutLayer for the second layer, and so on.
As long as you use the compatible regularizer associated with the different layers.

To launch standard variational dropout,
	
	python -um lasagne.experiments.mlp start-mlp \
		--input_directory=./mnist_784/ \
		--output_directory=./mnist_784/ \
		--minibatch_size=100 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9 \
		--dense_dimensions=100*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=VariationalDropoutLayer \
		--layer_activation_parameters=1.0*1.0 \
		--regularizer=kl_divergence_kingma:0.01,exponential_decay,1,0.9 \
		--debug=subsample_dataset

To launch sparse variational dropout,

	python -um lasagne.experiments.mlp start-mlp \
		--input_directory=./mnist_784/ \
		--output_directory=./mnist_784/ \
		--minibatch_size=100 \
		--number_of_epochs=10 \
		--learning_rate=0.01 \
		--dense_dimensions=100*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=SparseVariationalDropoutLayer \
		--layer_activation_parameters=0.8*0.5 \
		--regularizer=kl_divergence_sparse:0.01,exponential_decay,1,0.9 \
		--debug=subsample_dataset

Under any circumstances, you may also get help information and usage hints by adding `-h` or `--help` option.

## Start LeNet

To start a LeNet with two convolutional layers (with 4 and 8 ReLU filters each), with max-pooling after first convolutional layer and no pooling after second.
The network also includes one hidden layer of 32 ReLU neurons before output layer.

	python -um lasagne.experiments.lenet start-lenet \
		--input_directory=./mnist_1x28x28/ \
		--output_directory=./mnist_1x28x28/ \
		--minibatch_size=50 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9 \
		--convolution_filters=4*8 \
		--convolution_nonlinearities=rectify*rectify \
		--pool_mode=max*none \
		--dense_dimensions=32*10 \
		--dense_nonlinearities=rectify*softmax \
		--debug=subsample_dataset
		
To launch LeNet with adaptive dropout using Rademacher complexity regularization,

	python -um lasagne.experiments.lenet start-lenetA \
		--input_directory=./mnist_1x28x28/ \
		--output_directory=./mnist_1x28x28/ \
		--minibatch_size=50 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9 \
		--convolution_filters=4*8 \
		--convolution_nonlinearities=rectify*rectify \
		--dense_dimensions=32*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=AdaptiveDropoutLayer \
		--regularizer=rademacher:1e-10 \
		--debug=subsample_dataset

Similar commands to launch standard variational dropout on LeNet, 

	python -um lasagne.experiments.lenet start-lenet \
		--input_directory=./mnist_1x28x28/ \
		--output_directory=./mnist_1x28x28/ \
		--minibatch_size=50 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9 \
		--convolution_filters=4*8 \
		--convolution_nonlinearities=rectify*rectify \
		--dense_dimensions=32*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=VariationalDropoutLayer \
		--regularizer=kl_divergence_kingma:0.01,exponential_decay,1,0.9 \
		--debug=subsample_dataset

Similar commands to launch sparse variational dropout on LeNet

	python -um lasagne.experiments.lenet start-lenet \
		--input_directory=./mnist_1x28x28/ \
		--output_directory=./mnist_1x28x28/ \
		--minibatch_size=50 \
		--number_of_epochs=10 \
		--learning_rate=0.01,exponential_decay,1,0.9 \
		--convolution_filters=4*8 \
		--convolution_nonlinearities=rectify*rectify \
		--dense_dimensions=32*10 \
		--dense_nonlinearities=rectify*softmax \
		--layer_activation_types=SparseVariationalDropoutLayer \
		--regularizer=kl_divergence_sparse:0.01,exponential_decay,1,0.9 \
		--debug=subsample_dataset

Under any circumstances, you may also get help information and usage hints by adding `-h` or `--help` option.

## Start RNN

To start an Elman RNN with two recurrent layers.
The network starts with an embedding layer of 200 dimension.

	python -um lasagne.experiments.elman start-elman \
		--input_directory=./ptb_20x1x1/ \
		--output_directory=./ptb_20x1x1/ \
		--minibatch_size=100 \
		--number_of_epochs=15 \
		--embedding_dimension=200 \
		--learning_rate=1.0,exponential_decay,0.5,4 \
		--layer_dimensions=[200*200]*10000 \
		--layer_nonlinearities=[tanh*tanh]*softmax \
		--debug=subsample_dataset
	
Under any circumstances, you may also get help information and usage hints by adding `-h` or `--help` option.
