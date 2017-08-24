import numpy
import theano

__all__ = [
	"constant",
	"piecewise_constant",
	"exponential_decay",
	"natural_exp_decay",
	"inverse_time_decay"
]


def constant(learning_rate):
	return numpy.asarray(learning_rate).astype(theano.config.floatX)


def piecewise_constant(learning_rate, global_step, boundaries, values):
	"""Applies piecewise constant decay to the learning rate.

	Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5
	  for steps 100001 to 110000, and 0.1 for any additional steps.
	"""
	if global_step < boundaries[0]:
		return numpy.asarray(learning_rate).astype(theano.config.floatX)
	for low, high, v in zip(boundaries[:-1], boundaries[1:], values[:-1]):
		if global_step >= low and global_step < high:
			return numpy.asarray(v).astype(theano.config.floatX)
	return numpy.asarray(values[-1]).astype(theano.config.floatX)


def polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=1e-4, power=1.0, cycle=False):
	"""Applies a polynomial decay to the learning rate.

	```python
	global_step = min(global_step, decay_steps)
	decayed_learning_rate = (learning_rate - end_learning_rate) *
	                      (1 - global_step / decay_steps) ^ (power) +
	                      end_learning_rate
	```
	If `cycle` is True then a multiple of `decay_steps` is used, the first one
	that is bigger than `global_steps`.
	```python
	decay_steps = decay_steps * ceil(global_step / decay_steps)
	decayed_learning_rate = (learning_rate - end_learning_rate) *
	                      (1 - global_step / decay_steps) ^ (power) +
	                      end_learning_rate
	```
	"""
	if cycle:
		# Find the first multiple of decay_steps that is bigger than global_step.
		decay_steps = decay_steps * numpy.ceil(global_step / decay_steps)
	else:
		# Make sure that the global_step used is not bigger than decay_steps.
		global_step = numpy.minimum(global_step, decay_steps)

	progress = 1.0 * global_step / decay_steps
	return ((learning_rate - end_learning_rate) * numpy.power(1 - progress, power) + end_learning_rate).astype(
		theano.config.floatX)


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate, decay_delay=0, staircase=False):
	"""Applies exponential decay to the learning rate.

	decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

	"""
	adjusted_global_step = 0 if global_step < decay_delay else global_step - decay_delay
	adjusted_global_step = 1.0 * adjusted_global_step / decay_steps
	if staircase:
		adjusted_global_step = numpy.floor(adjusted_global_step)

	return (learning_rate * numpy.power(decay_rate, adjusted_global_step)).astype(theano.config.floatX)


def natural_exp_decay(learning_rate, global_step, decay_steps, decay_rate, decay_delay=0, staircase=False):
	"""Applies natural exponential decay to the learning rate.

	decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
	"""
	adjusted_global_step = 0 if global_step < decay_delay else global_step - decay_delay
	adjusted_global_step = 1.0 * adjusted_global_step / decay_steps
	if staircase:
		adjusted_global_step = numpy.floor(adjusted_global_step)

	return (learning_rate * numpy.exp(-decay_rate * adjusted_global_step)).astype(theano.config.floatX)


def inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, decay_delay=0, staircase=False):
	"""Applies inverse time decay to the learning rate.

	decayed_learning_rate = learning_rate / (1 + decay_rate * t)

    learning_rate: A Python number.
    	Initial learning rate.
    epoch_index: A Python number.
		Global step to use for the decay computation.  Must not be negative.
    decay_rate: A Python number.
        The decay rate.
    decay_steps: How often to apply decay.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'InverseTimeDecay'.

	Returns:
        The decayed learning rate.
	"""
	adjusted_global_step = 0 if global_step < decay_delay else global_step - decay_delay
	adjusted_global_step = 1.0 * adjusted_global_step / decay_steps
	if staircase:
		adjusted_global_step = numpy.floor(adjusted_global_step)

	return numpy.asarray(learning_rate / (1 + decay_rate * adjusted_global_step)).astype(theano.config.floatX)


def main():
	for x in xrange(10):
		print x, exponential_decay(1, x, 5, 0.1)


if __name__ == '__main__':
	main()
