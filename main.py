import numpy as np
import tensorflow as tf


#define the axis/dimension along to be normalized
AXIS = 0

# example inputs, uncomment one of them
example_tensor  = tf.constant( np.arange(3), dtype=tf.float64 )
#example_tensor = tf.constant( np.arange(4).reshape(2, 2), dtype=tf.float64 ) 
#example_tensor = tf.constant( np.arange(8).reshape(2, 2, 2), dtype=tf.float64 )

# defining my operation - normalization along the specific dimension
my_op = tf.nn.softmax(example_tensor, axis=AXIS)


with tf.Session() as sess:
	print("Input tensor with shape {}:".format(example_tensor.shape))
	print( sess.run(example_tensor) )

	print("\nNormalized output tensor along the axis {}:".format(AXIS))

	try:
		print( sess.run(my_op) )
	except tf.errors.InvalidArgumentError:
		print("\tThe tensor cannot be normalized along the dimension/axis " + str(AXIS) + ", possible dimensions/axes are 0-" + str(len(example_tensor.shape) - 1) + ".")


