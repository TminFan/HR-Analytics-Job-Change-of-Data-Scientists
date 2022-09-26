"""
This module is to create a deep neural network model.

Class:
    SimpleDnn

Functions:
    __init__
    deep_neural_network
"""

import tensorflow as tf

class SimpleDnn():
    """A class for building a deep neural network."""
    def __init__(self) -> None:
        pass

    def deep_neural_network(self, input_ds, layers_neurons_list, output_bias=None,
                        regularizer_lambda=0.01, regularizer="l1"):
        """
        This function is create a deep neural network.

        Args:
        input_ds: a numpy array. Shape is (datapoint_nums, features).
        layers_neurons_list: a Python list containing each hidden layers' numbers of neurons.
        regularizer: by default is l1, but can choose l2 also.
        regularizer_lambda: by default is 0.01, but can choose other rates too.
        """
        tf.random.set_seed(32)
        feature_nums = input_ds.shape[-1]
        initial_input = tf.keras.Input(shape=(feature_nums,), name="initial_input")
        layer_input = initial_input

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        for layer_index, layer_neurons in enumerate(layers_neurons_list):

            final_layer_index = len(layers_neurons_list) - 1
            if layer_index == final_layer_index:
                final_layer_output = tf.keras.layers.Dense(units=layer_neurons,
                                                            activation="sigmoid",
                                                            bias_initializer=output_bias
                                                            )(layer_input)
            else:
                if regularizer == "no_regularizer":
                    hidden_layer_output = tf.keras.layers.Dense(units=layer_neurons,
                                                                activation="relu")(layer_input)
                elif regularizer == "l1":
                    hidden_layer_output = tf.keras.layers.Dense(
                        units=layer_neurons,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L1(l1=regularizer_lambda)
                        )(layer_input)
                elif regularizer == "l2":
                    hidden_layer_output = tf.keras.layers.Dense(
                        units=layer_neurons,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L2(l2=regularizer_lambda)
                        )(layer_input)

            layer_input = hidden_layer_output

        model = tf.keras.Model(inputs=initial_input, outputs=final_layer_output)

        return model
