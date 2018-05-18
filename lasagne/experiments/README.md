# Lasagne/lasagne/experiments

This is an extension of [Lasagne](https://github.com/Lasagne/Lasagne) with a more user-friendly API.

The package includes a pre-processed copy of [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for experimental purpose.
Each image is reshaped to a 784 dimension float vector, and all pixels are unitized to a float value between 0 and 1.  

The package also features an implementation of [adaptive dropout learning](https://www.microsoft.com/en-us/research/publication/adaptive-dropout-rademacher-complexity-regularization/) as described in 

> Ke Zhai and Huan Wang, [Adaptive Dropout with Rademacher Complexity Regularization](https://openreview.net/pdf?id=S1uxsye0Z), International Conference on Learning Representations (ICLR-2018), Apr 2018.

Please send any bugs or problems to [Ke Zhai](<FIRST><LAST>@microsoft.com).

## Launch and Execute

Assume the Lasagne package is downloaded to directory `$PROJECT_SPACE/`, i.e., 

	$PROJECT_SPACE/Lasagne/

To prepare the example MNIST dataset,

	cd $PROJECT_SPACE/Lasagne/
	tar zxvf data/mnist.tar.gz
	
You should see the data directory called `mnist/` containing the `[test|train].[feature|label].npy` files.

To start a multi-layer perceptron with one hidden layer of 100 ReLU neurons,
	
	python -um lasagne.experiments.mlp start-mlp \
		--input_directory=./mnist/ \
		--output_directory=./mnist/ \
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
		--input_directory=./mnist/ \
		--output_directory=./mnist/ \
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
The `snapshot` option specifies the snapshot function to store the intermediate states during training, defined in [this file](../experiments/debug.py). 

To launch standard variational dropout,
	
	python -um lasagne.experiments.mlp start-mlp \
		--input_directory=./mnist/ \
		--output_directory=./mnist/ \
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
		--input_directory=./mnist/ \
		--output_directory=./mnist/ \
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
