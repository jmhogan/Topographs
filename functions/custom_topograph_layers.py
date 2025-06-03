"""
Custom layers and functions for the topograph model.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .custom_base_layers import FCDenseBlock
from .custom_graph_layers import (
    AttentionLayer,
    EdgeAggregation,
    EdgeBlock,
    PoolingLayer,
    expand_edges_with_mask,
    pass_with_mask,
)


@dataclass
class TopoGraphData:
    """
    Class to hold all data needed for the topograph model, containing jets, a mask for
    existing/non-existing jets, helper nodes for ws and tops, and previous edges.
    """

    jets: tf.Tensor
    mask: tf.Tensor
    nodes_top: tf.Tensor = None
    previous_edges: List[tf.Tensor] = None


class ParticlePredictionMLPs(Layer):
    """
    Create MLPs for regression for multiple particles of the same type at the same time.
    """

    def __init__(self, architecture: dict, n_particles: int = 2, **kwargs):
        """
        Create MLPs for regression for multiple particles of the same type at the same
        time. For each particle a separate MLP with the same HPs will be created and
        used for regression.

        Args
        ----
            architecture:
                Dictionary holding the configuration of the MLPs. To be used with
                'FCDenseBlock'. Additionally, an 'out' key is needed to specify the
                number of outputs of the regression networks. No activation is used for
                the final linear layer.
            n_particles:
                Number of particles. That many MLPs will be created.

        """
        super().__init__(**kwargs)
        self.n_particles = n_particles
        self.dense_blocks = [
            FCDenseBlock(
                architecture, {"units": architecture["out"], "activation": "linear"}
            )
            for _ in range(self.n_particles)
        ]

    def call(self, tns: List[tf.Tensor]) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            tns:
                List of tensors where each tensor corresponds to one particle/node, i.e.
                the length of the list should be 'self.n_particles'.
                Expected shape of each tensor: (batch_size, n_input_features).
                Additional dimensions can be included.

        Returns
        -------
            Regression results for all particles concatenated into one tensor.
            Expected shape: (batch_size, n_particles, architecture['out']). Additional
            dimensions can be included.

        """
        outputs = []
        for i, dense_block in enumerate(self.dense_blocks):
            dense = dense_block(tns[:, i])
            outputs.append(tf.expand_dims(dense, 1))

        return tf.concat(outputs, axis=1)

class InitializeHelperNodesTopFixed(Layer):
    """
    Initialize the helper nodes representing the top bosons from the jets and the W
    nodes.
    """

    def __init__(
        self,
        jets_pooling: str,
        attention_net_architecture: Optional[dict],
        **kwargs,
    ):
        """
        Initialize the helper nodes representing the top bosons from the jets and the W
        nodes. The jets are pooled in some configurable way and the W node is
        concatenated onto it. This fixes a connection between the W nodes and the top
        nodes.

        Args
        ----
            jets_pooling:
                Pooling to be applied to the jets to get part of the to be initialized
                helper nodes.
            attention_net_architecture:
                If attention pooling is wanted, architecture of the attention network

        """
        super().__init__(**kwargs)

        if jets_pooling == "att":
            self.first_attention_net = AttentionLayer(attention_net_architecture)
            self.second_attention_net = AttentionLayer(attention_net_architecture)
        # Pooling layers don't have parameters, so the same can be used for both tops
        self.jets_pooling = jets_pooling
        self.jets_pooling_layer = PoolingLayer(jets_pooling)
        
        self.top_seed_net = FCDenseBlock(
            {"units": [64, 32], "activation": "relu", "dropout": 0.1},
            {"units": 32, "activation": "linear"}
        )
        

    def call(self, inputs: List[tf.Tensor], mask: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                List of the inputs to initialize the helper nodes. The first entry is
                for the jet nodes and the second for the W nodes.
                Expected shape: [(batch_size, n_nodes, n_features_1),
                                 (batch_size, 2, n_features_2)]
            mask:
                Mask for existing/non-existing nodes/jets.
                Expected shape: (batch_size, n_nodes, 1)

        Returns
        -------
            Initialized helper nodes.
            Expected shape: (batch_size, 2, n_features_1 + n_features_2)

        """
        if hasattr(self, "first_attention_net"):
            first_attention_weights = self.first_attention_net(inputs[0], mask)
            second_attention_weights = self.second_attention_net(inputs[0], mask)

            first_pooled_jets = self.jets_pooling_layer(
                [inputs[0], first_attention_weights], mask
            )
            second_pooled_jets = self.jets_pooling_layer(
                [inputs[0], second_attention_weights], mask
            )
        else:
            first_pooled_jets = self.jets_pooling_layer(inputs[0], mask)
            second_pooled_jets = self.jets_pooling_layer(inputs[0], mask)
        
        # Process the pooled representation to better represent a top quark
        # that directly decays to 3 jets
        first_top = self.top_seed_net(first_pooled_jets)
        second_top = self.top_seed_net(second_pooled_jets)
        #first_top = tf.concat([first_pooled_jets], axis=-1) #, inputs[1][:, 0]]
        #second_top = tf.concat([second_pooled_jets], axis=-1) #, inputs[1][:, 1]]

        return tf.concat([first_top[:, None, :], second_top[:, None, :]], axis=1)
       

class EdgeClassification(Layer):
    """
    Form edges between nodes and classify them.
    """

    def __init__(self, classification_net: dict, **kwargs):
        """
        Form edges between nodes and classify them.

        Args
        ----
            classification_net:
                Architecture of the network to classify the edges. A final layer with
                one output and a sigmoid activation is added at the end.
            n_jets_per_top:
                Number of jets per top quark (default: 3)

        """
        super().__init__(**kwargs)
        self.classification_net = classification_net
        self.edge_aggregation = EdgeAggregation(fully_connected=True)
        self.classification_net = FCDenseBlock(
            classification_net, {"units": 1, "activation": "sigmoid"}
        )

    def call(self, inputs: List[tf.Tensor], mask: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                Nodes between which the edges should be classified. The first entry
                in the list should be the jet nodes and the second the W/top nodes.
                Expected shape: [(batch_size, n_nodes_type_1, n_features_1),
                                 (batch_size, n_nodes_type_2, n_features_2)]
                                 (n_features_i doesn't have any impact on this
                                 implementation)

            mask:
                Mask of existing and non-existing nodes for the first entry in the input
                list. It is assumed that all nodes for the second entry in the input
                list exist.
                Expected shape: (batch_size, n_nodes_type_1, 1)

        Returns
        -------
            edge_scores:
                Score for every edge between the two types of nodes.
                Expected shape: (batch_size, n_nodes_type_1, n_nodes_type_2, 1)

        """
        edges_jet_particle, mask_jet_particle = self.edge_aggregation(
            inputs[1], inputs[0], None, mask
        )

        edge_scores = self.classification_net(edges_jet_particle)
        edge_scores = expand_edges_with_mask(edge_scores, mask_jet_particle)
        

        return edge_scores


class TopoGraphBlock(Layer):
    """
    Class defining a Topograph block.
    """

    def __init__(
        self,
        topo_graph_layer_kwargs: dict,
        classification_net: dict,
        regression_net: dict,
        **kwargs,
    ):
        """
        Class defining a Topograph block: a TopographLayer for information exchange
        between all nodes, two EdgeClassification layers to classify edges to the W and
        top nodes, and two ParticlePredictionMLPs to regress towards the true parton
        properties for the W and top nodes.

        Args
        ----
            topo_graph_layer_kwargs:
                Configuration for the TopoGraphLayer in dictionary form.
            classification_net:
                Configuration for the edge classification networks. All edge
                classification networks in one block have the same HPs.
            regression_net:
                Configuration for the regression networks. All regression networks in
                one block have the same HPs.

        """
        super().__init__(**kwargs)
        self.topo_graph_layer = TopoGraphLayer(**topo_graph_layer_kwargs)
        self.classification_top_layer = EdgeClassification(
            classification_net=classification_net
        )
        self.regression_top_layer = ParticlePredictionMLPs(
            architecture=regression_net, n_particles=2
        )

    def call(
        self, data: TopoGraphData
    ) -> Tuple[TopoGraphData, tf.Tensor, tf.Tensor]:
        """
        Call the layer

        Args
        ----
            data:
                TopoGraphData holding the jet nodes, top nodes, mask for
                existing/non-existing jets, and potentially previous edge features.

        Returns
        -------
            data:
                Updated TopographData. Will update everything apart from the mask.
            regression_top:
                Regression results for the two top nodes.
                Expected shape: (batch_size, 2, n_regression_features)
            classification_top:
                Edge classification results for the two top nodes.
                Expected shape: (batch_size, n_jets, 2, 1)

        """
        data = self.topo_graph_layer(data)
        regression_top = self.regression_top_layer(data.nodes_top)
        classification_top = self.classification_top_layer(
            [data.jets, data.nodes_top], data.mask
        )

        return data, regression_top, classification_top


class TopoGraphLayer(Layer):
    """
    A single TopoGraphLayer which passes messages between all different types of nodes.
    """

    def __init__(
        self,
        edge_net_architecture: dict,
        node_net_architecture: dict,
        pooling: str = "avg",
        attention_net_architecture: Optional[dict] = None,
        full_connections_jets: bool = False,
        full_connections_tops: bool = False,
        top_top_interaction: bool = False,
        **kwargs,
    ):
        """
        A single TopoGraphLayer which passes messages between all different types of
        nodes.

        Args
        ----
            edge_net_architecture:
                Architecture to process edges. A separate network is built for each type
                of edge (e.g. jet-W and W-jet are different edges). All edge networks
                have the same HPs.
            node_net_architecture:
                Architecture to process nodes after they have been updated with the
                pooled edges. A separate network is built for each type of node. All
                node networks have the same HPs.
            pooling:
                Pooling operation used to pool edges.
            attention_net_architecture:
                Configuration of the attention network if attention pooling is
                requested. A separate network is built for each type
                of edge. All attention networks have the same HPs.
            full_connection_jets:
                Connect all jets with each other, including self connections.
            full_connections_tops:
                Connect all Tops with each other, including self connections.
            top_top_interaction:
                Let the two Top nodes interact with each other.

        """
        super().__init__(**kwargs)

        # TODO: Might need to modify this

        self.jet_jet_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            k_neighbours=15,
            fully_connected=full_connections_jets,
        )
        
        self.jet_top_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )

        self.dense_jets = FCDenseBlock(node_net_architecture)

        self.top_jet_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )
        
        if top_top_interaction:
            self.top_top_block = EdgeBlock(
                dense_edges=edge_net_architecture,
                pooling_edges="avg",
                attention_network=None,
                k_neighbours=1,
                fully_connected=full_connections_tops,
            )

        self.dense_tops = FCDenseBlock(node_net_architecture)

    def call(self, data: TopoGraphData) -> TopoGraphData:
        """
        Call the layer

        Args
        ----
            data:
                TopoGraphData holding jet nodes, W nodes, top nodes, a mask for
                existing/non-existing jets, and potentially previous edges.

        Returns
        -------
            updated_data:
                Updated TopoGraphData. Everything except the mask is updated.

        """
        updated_data = TopoGraphData(
            jets=None, mask=data.mask, previous_edges=[None] * 4
        )

        (pooled_jet_jet_edges, updated_data.previous_edges[0],) = self.jet_jet_block(
            data.jets, data.jets, data.mask, data.mask, data.previous_edges[0]
        )
    
        (pooled_jet_top_edges, updated_data.previous_edges[1],) = self.jet_top_block(
            data.jets, data.nodes_top, data.mask, None, data.previous_edges[1]
        )

        updated_jets = tf.concat(
            [data.jets, pooled_jet_jet_edges, pooled_jet_top_edges],
            axis=-1,
        )
        updated_data.jets = pass_with_mask(updated_jets, self.dense_jets, data.mask)

        (pooled_top_jet_edges, updated_data.previous_edges[2],) = self.top_jet_block(
            data.nodes_top, data.jets, None, data.mask, data.previous_edges[2]
        )
        
        if hasattr(self, "top_top_block"):
            (
                pooled_top_top_edges,
                updated_data.previous_edges[3],
            ) = self.top_top_block(
                data.nodes_top, data.nodes_top, None, None, data.previous_edges[3]
            )
            updated_tops = tf.concat(
                [
                    data.nodes_top,
                    pooled_top_jet_edges,
                    pooled_top_top_edges,
                ],
                axis=-1,
            )
        else:
            updated_tops = tf.concat(
                [data.nodes_top, pooled_top_jet_edges], axis=-1 #, pooled_top_w_edges
            )
        updated_data.nodes_top = self.dense_tops(updated_tops)

        return updated_data
