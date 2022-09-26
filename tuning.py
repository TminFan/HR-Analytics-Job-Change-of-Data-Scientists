"""
It is a hyperparameters tuning module for dnn_model1 and dnn_model2.
It automatically tunes optimizers and regularizer lambda rates.
For regularizer, it needs to specify whether it is l1 or l2.

Class:
    GridSearch

Functions:
    __init__
    create_hparams
    train_test_model
    run
    start_run
"""

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from dnn_model1 import DnnWithEmbeddingLayers
from dnn_model2 import SimpleDnn

class GridSearch():
    """
    Tuning hyperparameters of dnn_model1 and dnn_model2 with TensorBoard plugin tools.
    It automatically tunes optimizers and regularizer lambda rates.
    For regularizer, it needs to specify whether it is l1 or l2.
    """
    def __init__(self, run_dir, metrics_name_list, model_name, x_train, y_train, x_test, y_test,
                metrics, epochs, layer_neurons_list,
                embedding_col_nums=None, embedding_cols_dataset_indexes=None,
                embedding_cols_output_dims=None, initial_bias=None,
                classweight=None, regularizer=None) -> None:
        """Initialize attributes"""
        self.run_dir = run_dir
        # metrics_name_list is a list of metrics' names.
        self.metrics_name_list = metrics_name_list
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # metrics is a list of metrics functions that are used in model compile.
        self.metrics = metrics
        self.epochs = epochs
        self.layer_neurons_list = layer_neurons_list
        self.embedding_col_nums = embedding_col_nums
        self.embedding_cols_dataset_indexes = embedding_cols_dataset_indexes
        self.embedding_cols_output_dims = embedding_cols_output_dims
        self.initial_bias = initial_bias
        self.classweight = classweight
        self.regularizer = regularizer

    def create_hparams(self, optimizer_list, regularizer_lambdarates):
        """
        Create hyperparameters of optimizer and regularizer that will be used in tuning.
        If regularizer is specified as l1 at initialization, then self.hp_regul is a l1 regularizer
        with your choice of lambda rates. So does regularizer specified as l2.

        Args:
        optimizer_list: a list of optimizers. e.g. ["adam", "rmsprop"]
        regularizer_lambdarates: a list of lambda rates of regularizer. e.g. [0.01, 0.03]
        """
        self.hp_optimizer = hp.HParam("optimizer", hp.Discrete(optimizer_list))
        if self.regularizer=="l1":
            self.hp_regul = hp.HParam("l1 regularizer", hp.Discrete(regularizer_lambdarates))
        elif self.regularizer=="l2":
            self.hp_regul = hp.HParam("l2 regularizer", hp.Discrete(regularizer_lambdarates))

        return self.hp_optimizer, self.hp_regul

    def train_test_model(self, hparams):
        """Create a function that includes training and testing model.

        hparams: it is a dictionary that will be created in below StartRun function.
                It includes all hyperparameter values.
                for example: {hp_regul: l1, hp_optimizer: optimizer}
        """
        if self.model_name=="model1":

            model_generator = DnnWithEmbeddingLayers()

            model = model_generator.deep_neural_network(
                self.x_train, self.embedding_col_nums,
                self.embedding_cols_dataset_indexes,
                self.embedding_cols_output_dims,
                self.layer_neurons_list, output_bias=self.initial_bias,
                regularizer=self.regularizer
            )

            # Every run will use different optimizer function in hparams[self.hp_optimizer].
            model.compile(optimizer=hparams[self.hp_optimizer],
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=self.metrics)

            model.fit(self.x_train, self.y_train, epochs=self.epochs, class_weight=self.classweight)

            result = model.evaluate(self.x_test, self.y_test)

        elif self.model_name=="model2":

            model_generator = SimpleDnn()

            model = model_generator.deep_neural_network(
                self.x_train, self.layer_neurons_list,
                output_bias=self.initial_bias, regularizer=self.regularizer
            )
            # Every run will use different optimizer function in hparams[self.hp_optimizer].
            model.compile(optimizer=hparams[self.hp_optimizer],
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=self.metrics)

            model.fit(self.x_train, self.y_train, epochs=self.epochs, class_weight=self.classweight)

            result = model.evaluate(self.x_test, self.y_test)

        model_measures = result

        return model_measures

    def run(self, hparams):
        """
        For each run, log an hparams summary with the hyperparameters and final metrics.

        hparams: it is a dictionary that will be created in below StartRun function.
                It includes all hyperparameter values.
                for example: {hp_regul: l1, hp_optimizer: optimizer}
        """

        with tf.summary.create_file_writer(self.run_dir + self.run_name).as_default():

            hp.hparams(hparams)
            model_measures = self.train_test_model(hparams)
            for metric_idx, metrics_name in enumerate(self.metrics_name_list):
                tf.summary.scalar(
                    metrics_name,
                    model_measures[metric_idx+1], step=self.epochs
                )

    def start_run(self):
        """Start running the tuning."""
        session_num = 0

        if self.regularizer=="l1":
            for l1 in self.hp_regul.domain.values:
                for optimizer in self.hp_optimizer.domain.values:
                    hparams = {
                        self.hp_regul: l1,
                        self.hp_optimizer: optimizer
                    }
                    self.run_name = "run-%d" % session_num
                    print("--- Starting trial: %s" % self.run_name)
                    print({h.name: hparams[h] for h in hparams})
                    self.run(hparams)
                    session_num += 1
        elif self.regularizer=="l2":
            for l2 in self.hp_regul.domain.values:
                for optimizer in self.hp_optimizer.domain.values:
                    hparams = {
                        self.hp_regul: l2,
                        self.hp_optimizer: optimizer
                    }
                    self.run_name = "run-%d" % session_num
                    print("--- Starting trial: %s" % self.run_name)
                    print({h.name: hparams[h] for h in hparams})
                    self.run(hparams)
                    session_num += 1

        return "Done"
