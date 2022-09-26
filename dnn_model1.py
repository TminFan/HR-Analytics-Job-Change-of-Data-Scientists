"""
This module is to create a deep neural network model
with embedding layers that are used for preprocessing categorical data features.

Class:
    DNNWithEmbeddingLayers

Functions:
    __init__
    deep_neural_network
"""

import numpy as np
import tensorflow as tf

class DnnWithEmbeddingLayers():
    """A class for building a deep neural network."""
    def __init__(self) -> None:
        pass

    def deep_neural_network(self, input_ds, embedding_col_nums, embedding_cols_dataset_indexes,
                        embedding_cols_output_dims, layers_neurons_list, output_bias=None,
                        regularizer_lambda=0.01, regularizer="l1"):
        """
        Create a deep neural network model with embedding layers that are used for preprocessing
        categorical data features.

        Args:
        input_ds: a numpy array. Shape is (datapoint_nums, features).
        embedding_col_nums: numbers of columns that will be embedded to vectors.
        embedding_cols_dataset_indexes: a nested list containing the column position indexes in
                                        the dataframe.
                                        for example [[44, 45], [45, 46]].
                                        44 is 44th column in the dataframe.
        embedding_cols_output_dims: a list containing output_dims for each embedding columns.
                                    Length of this list should be the same as embedding_col_nums.
        layers_neurons_list: a list containixng each hidden layers' numbers of neurons.
        regularizer: by default is L1, but can choose L2 too.
        regularizer_lambda: by default is 0.01.
        """
        tf.random.set_seed(42)
        feature_nums = input_ds.shape[-1]
        initial_input = tf.keras.Input(shape=(feature_nums,), name="initial_input")

        embedding_layers_output = []
        for embedding_col_index in range(embedding_col_nums):

            dataframe_start_index = embedding_cols_dataset_indexes[embedding_col_index][0]
            dataframe_end_index = embedding_cols_dataset_indexes[embedding_col_index][1]
            unique_values = np.unique(input_ds[:, dataframe_start_index : dataframe_end_index])
            embedding_input_dims = unique_values.shape[0]
            embedding_output_dims = embedding_cols_output_dims[embedding_col_index]

            embedding_input = initial_input[:, dataframe_start_index : dataframe_end_index]
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=embedding_input_dims,
                output_dim=embedding_output_dims,
                name=f"embedding_layer{embedding_col_index}"
            )
            # embedding_output shape is (datapoint_nums, 1, embedding_output_dims).
            embedding_output = embedding_layer(embedding_input)
            # flatten_embedding_output shape is (datapoint_nums, embedding_output_dims)
            flatten_embedding_output = tf.keras.layers.Flatten()(embedding_output)
            embedding_layers_output.append(flatten_embedding_output)

        embedding_layers_output.append(initial_input[:, :embedding_cols_dataset_indexes[0][0]])
        concat_input = tf.keras.layers.Concatenate(axis=1)(embedding_layers_output)

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        layer_input = concat_input
        for layer_index, layer_neurons in enumerate(layers_neurons_list):

            final_layer_index = len(layers_neurons_list) - 1
            if layer_index == final_layer_index:
                final_layer_output = tf.keras.layers.Dense(units=layer_neurons,
                                                        activation="sigmoid",
                                                        bias_initializer=output_bias)(layer_input)
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
        