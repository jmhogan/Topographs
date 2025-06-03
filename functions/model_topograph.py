"""
Full Topograph model
"""
from typing import List

import tensorflow as tf

from .custom_graph_layers import CreateMask, GraphBlock
from .custom_losses import ClassificationLoss, RegressionLoss
from .custom_topograph_layers import (
    InitializeHelperNodesTopFixed,
    ParticlePredictionMLPs,
    TopoGraphBlock,
    TopoGraphData,
)
from .model_base import ModelBaseClass
from .tools import Dataset

# Modifications:
#    Changing Indices (I hope those are right)
#    Removed zip in calculate_regression_loss


class TopographModel(ModelBaseClass):
    """
    Full Topograph model
    """

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self._build(config)
        self._create_metrics()

        self.regression_loss = RegressionLoss(config["regression_loss"])

        self.classification_loss = ClassificationLoss(
            **config["classification_loss"],
        )
        self.persistent_edges = config["persistent_edges"]
        self.i = 0

    def _create_metrics(self) -> None:
        """
        Create metrics for all outputs. These include regression metrics for the
        initialisation and one after each topograph block, and one classification metric
        after each topograph block. The actual value is not calculated by these members.
        The value of the respective metric will be used to update the state of these
        metrics to keep all the nice tf metric functionality.
        """
        self.metric_initialisation = tf.keras.metrics.Mean("Initialisation")
        self.metric_regression = [
            tf.keras.metrics.Mean(f"Regression_{i}") for i in range(self.n_scores)
        ]
        self.metric_classification = [
            tf.keras.metrics.Mean(f"Classification_{i}") for i in range(self.n_scores)
        ]

        self.my_metrics = (
            [self.metric_initialisation]
            + self.metric_regression
            + self.metric_classification
        )

    def _build(self, config: dict) -> None:
        """
        Create all layers that are needed for the model based on some configuration
        dictionary.
        """
        self.create_mask = CreateMask()

        ########################################################################
        # Initial graph block, jets exchange information and get updated
        ########################################################################
        n_iterations = config["initial_graph_block"].pop("n_iterations")
        self.jet_graph_block = [
            GraphBlock(**config["initial_graph_block"]) for _ in range(n_iterations)
        ]

        ########################################################################
        # Top initialization block, tops get initialized and a regression network
        # for the initialized values is used
        ########################################################################
        initialization_top = config["initialization_top"]
        self.initialize_top = InitializeHelperNodesTopFixed(
            jets_pooling=initialization_top["jets_pooling"],
            attention_net_architecture=initialization_top["attention_net_architecture"],
        )
        self.initial_top_regression = ParticlePredictionMLPs(
            architecture=initialization_top["regression_net"], n_particles=2
        )

        ########################################################################
        # TopoGraph layers
        ########################################################################
        self.n_scores = config["Topograph"].pop("n_iterations")
        self.topo_blocks = [
            TopoGraphBlock(
                config["Topograph"],
                config["edge_classification"],
                config["regression_net"],
            )
            for _ in range(self.n_scores)
        ]

    def call(self, inputs: List[tf.Tensor]):
        """
        One pass through the model.
        """
        data = TopoGraphData(jets=inputs[0], mask=self.create_mask(inputs[0]))

        for layer in self.jet_graph_block:
            data.jets = layer(data.jets, data.mask)
        
        data.nodes_top = self.initialize_top([data.jets], data.mask)
        initialised_nodes_top = self.initial_top_regression(data.nodes_top)

        data.previous_edges = [None] * 4 
        (
            regression_loss_top,
            classification_loss_top,
        ) = ([], [])
        for block in self.topo_blocks:
            (
                data,
                regression_top,
                classification_top,
            ) = block(data)
            regression_loss_top.append(regression_top)
            classification_loss_top.append(classification_top)
            if not self.persistent_edges:
                data.previous_edges = [None] * 4 
        
        file = open(f"inputs[1].txt", "w+")
        content = str(inputs[1])
        file.write(content)
        file.close()
        
        file = open(f"datamask.txt", "w+")
        content = str(data.mask)
        file.write(content)
        file.close()
        
        file = open(f"jetsmask.txt", "w+")
        content = str(data.jets)
        file.write(content)
        file.close()
        
        return (
            initialised_nodes_top,
            regression_loss_top,
            classification_loss_top,
            data.mask,
            inputs[1],
        )

    def custom_fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        flavour_tagging: bool = True,
    ):
        """
        Build tf.data.Datasets for training and validation from the Dataset holding all
        the data and fit them.
        """
        train_x = tf.data.Dataset.from_tensor_slices(
            (train_dataset.jets.get_inputs(flavour_tagging), train_dataset.parton_mask)
        )
        train_y = tf.data.Dataset.from_tensor_slices(
            (
                train_dataset.true_edges_top,
                train_dataset.top_partons.momentum,
            )
        )
        val_x = tf.data.Dataset.from_tensor_slices(
            (val_dataset.jets.get_inputs(flavour_tagging), val_dataset.parton_mask)
        )
        val_y = tf.data.Dataset.from_tensor_slices(
            (
                val_dataset.true_edges_top,
                val_dataset.top_partons.momentum,
            )
        )

        train_dataset = tf.data.Dataset.zip((train_x, train_y))
        val_dataset = tf.data.Dataset.zip((val_x, val_y))

        self.custom_fit_datasets(train_dataset, val_dataset)

    def custom_fit_datasets(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
        """
        Fit the model using datasets by utilising the fit function of the base class.
        """
        super().custom_fit(train_ds, val_ds)

    def calculate_regression_loss(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, parton_mask: tf.Tensor
    ) -> List[tf.Tensor]:
        """
        Using the true and predicted values of the parton properties calculate all
        regression losses.
        """
        initialization_loss = self.regression_loss.calculate(
            y_true[1], y_pred[0], parton_mask 
        )[0]
        
        regression_losses = []
        for (preds_top) in y_pred[1]: 
            regression_losses.append(
                self.regression_loss.calculate(
                    y_true[1], preds_top, parton_mask
                )[0]
            )

        return [initialization_loss] + regression_losses

    def calculate_classification_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> List[tf.Tensor]:
        """
        Using the true and predicted values for all edges and the mask for the jets,
        calculate all classification losses.
        """
        classification_losses = []

        for (preds_top) in y_pred[2]:
            classification_losses.append(
                self.classification_loss.calculate(
                    y_true[0], preds_top, mask, parton_mask
                )
            )
        classification_losses[-1] = classification_losses[-1]

        return classification_losses

    def calculate_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculate both, regression and classification losses.
        """
        mask = mask[..., None]
        parton_mask = tf.expand_dims(parton_mask, -1)
        regression_losses = self.calculate_regression_loss(y_true, y_pred, parton_mask)
        classification_losses = self.calculate_classification_loss(
            y_true, y_pred, mask, parton_mask
        )

        loss = tf.reduce_sum(regression_losses) + tf.reduce_sum(classification_losses)

        return loss

    def calculate_regression_metrics(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, parton_mask: tf.Tensor
    ) -> None:
        """
        Update the state of all regression metrics.
        """

        self.metric_initialisation.update_state(
            self.regression_loss.calculate(
                y_true[1], y_pred[0], parton_mask 
            )[0]
        )

        for (metric, preds_top) in zip( 
            self.metric_regression, y_pred[1] 
        ):
            metric.update_state(
                self.regression_loss.calculate(
                    y_true[1], preds_top, parton_mask 
                )[0]
            )

    def calculate_classification_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> None:
        """
        Update the state of all classification metrics.
        """
        for (metric, preds_top) in zip( 
            self.metric_classification, y_pred[2] 
        ):
            metric.update_state(
                self.classification_loss.calculate(
                    y_true[0], preds_top, mask, parton_mask 
                )
            )

    def calculate_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> None:
        """
        Update the state of all metrics, regression and classification.
        """
        mask = mask[..., None]
        parton_mask = tf.expand_dims(parton_mask, -1)
        
        file = open(f"parton_mask.txt", "w+")
        content = str(parton_mask)
        file.write(content)
        file.close()
        
        file = open(f"maskMetrics.txt", "w+")
        content = str(mask)
        file.write(content)
        file.close()
        
        self.calculate_regression_metrics(y_true, y_pred, parton_mask)

        self.calculate_classification_metrics(y_true, y_pred, mask, parton_mask)
