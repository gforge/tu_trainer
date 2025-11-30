from __future__ import annotations
import traceback
from typing import Dict, TYPE_CHECKING
import pandas as pd
import time
import networkx as nx
from pydantic import BaseModel, ValidationError
from DataTypes import Modality_Cfg_Parser, Model_Cfg, Loss_Cfg_Parser
from Graphs.Losses.get_loss_type import get_loss_type

if TYPE_CHECKING:
    from Graphs.Losses.base_loss import Base_Loss

from file_manager import File_Manager
from UIs.console_UI import Console_UI
from UIs.Writers import AllWriters
from global_cfgs import Global_Cfgs
from .base_graph import Base_Graph
from .Models.model import Model


class Graph(Base_Graph):
    """ The Graph connects the networks into one structure

    Each network has heads & tails, heads are inputs and tail are outputs
    during enconding. During the decoding the opposite is true.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_frame = None
        self.time_frame_counter = 0
        self.__init_explicit_modalities()

        self.graph = self.__init_graph()
        self.graph_travers_order = self.__get_graph_traverse_order()

        self.models: Dict[str, Model] = {}
        self.__init_models_and_adjust_sizes()

        self.__init_remaining_modalities()

        self.losses: Dict[str, Base_Loss] = {}
        self.__init_losses()

        self._save_graph_to_mermaid_file()
        self.exception_counter = 0

        self.__save_train_visuals: bool = Global_Cfgs().get('save_train_visuals', default=False)

        self.__loss = 0
        self.__calculated_backwards = False

    def train(self, batch=None) -> bool:
        for model_name in self.graph_travers_order:
            self.models[model_name].train()
        for _, loss in self.losses.items():
            loss.train()

        if batch is None:
            return False

        try:
            self.encode(batch)
            if self.reconstruction:
                self.decode(batch)

            self.__loss += self.compute_loss(batch)

            if self.__save_train_visuals:
                AllWriters().add_last_data_2_visuals(batch=batch)

            self.exception_counter = 0
            return True
        except KeyError as e:
            Console_UI().warn_user(f'Could not find {e} in:')
            Console_UI().warn_user(sorted(batch.keys()))
            Console_UI().inform_user("\n\n Traceback: \n")
            traceback.print_exc()
            raise e
        except Exception as e:
            Console_UI().warn_user(f'** Error while training batch in {self.get_name()} **')
            Console_UI().warn_user(
                f'Indices: {batch["indices"]} and encoder image shape {batch["encoder_image"].shape}')
            Console_UI().warn_user(f'Error message: {e}')
            Console_UI().inform_user("\n\n Traceback: \n")
            traceback.print_exc()

            self.exception_counter += 1
            if self.exception_counter > 5:
                raise RuntimeError(f'Error during training: {e}')
            return False

    def backward(self) -> None:
        """Run backpropagation

        Run backprop if there is a loss saved and then take a step
        """
        if self.__loss == 0:
            self.__calculated_backwards = True
            return

        self.__loss.backward()
        self.__loss = 0
        self.__calculated_backwards = True

    def collect_runtime_stats(self, batch):
        start_time = batch['time'].pop('start')
        time_dict = {(outerKey, innerKey): values for outerKey, innerDict in batch['time'].items()
                     for innerKey, values in innerDict.items()}

        if self.time_frame is None:
            self.time_frame = pd.DataFrame(time_dict).T - start_time
            self.time_frame_counter = 1
        else:
            if self.time_frame_counter < 3:
                # just discard the first few samples, due to establishing connections
                # with GPU, they are not worth counting.
                self.time_frame = pd.DataFrame(time_dict).T - start_time
            else:
                self.time_frame *= self.time_frame_counter
                self.time_frame += pd.DataFrame(time_dict).T - start_time
                self.time_frame /= (self.time_frame_counter + 1)
            self.time_frame_counter += 1

        batch['time']['true_full_time'] = time.time() - start_time
        batch['time_stats'] = self.time_frame

        # Console_UI().debug(self.convert_to_mermaidjs(to_visualize='gantt'))

    def eval(self, batch=None):
        for model_name in self.graph_travers_order:
            self.models[model_name].eval()
        for _, loss in self.losses.items():
            loss.eval()
        if batch is not None:
            self.encode(batch)
            if self.reconstruction:
                self.decode(batch)
            self.compute_loss(batch)
            # The coordinates are calculated during the compute_loss()
            AllWriters().add_last_data_2_visuals(batch=batch)

    def encode(self, batch):
        for model_name in self.graph_travers_order:
            try:
                self.models[model_name].encode(batch)
            except RuntimeError as e:
                msg = f'Got error for {len(batch["indices"])} indices in model {model_name}:\n{e}\n' + \
                      f'\nThe encoder image shape {batch["encoder_image"].shape}'
                key = f'encoder_{model_name[:-5]}'
                if key in batch:
                    msg += f' where the shape of the input was {batch[key].shape}'
                else:
                    msg += f' but the input {key} could not be found :-('
                raise RuntimeError(msg)

    def decode(self, batch):
        for model_name in reversed(self.graph_travers_order):
            self.models[model_name].decode(batch)

    def compute_loss(self, batch, accumulated_loss=0):
        for loss in self.losses.values():
            accumulated_loss += loss(batch) * loss.coef
        return accumulated_loss

    def step(self):
        """Updates weights

        Activate optimizer for each neural network and takes a step
        according to the loss function.
        """
        if not self.__calculated_backwards:
            return

        for model_name in self.graph_travers_order:
            self.models[model_name].step()

        # Losses can have their own neural_network
        for loss in self.losses.values():
            loss.step()

        self.__calculated_backwards = False

    def zero_grad(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].zero_grad()
        for _, loss in self.losses.items():
            loss.zero_grad()

    def save(self, scene_name):
        Console_UI().inform_user('\n*****************************************' +
                                 f'\nSave network and losses for {scene_name}')

        # If we have backprop not at every iteration we want to make sure that the backprop
        # is saved before storing the network.
        if self.__loss:
            self.backward()
            self.step()

        no_networks = 0
        no_losses = 0
        for model_name in self.graph_travers_order:
            self.models[model_name].save(scene_name)
            no_networks += 1
        for _, loss in self.losses.items():
            loss.save(scene_name)
            no_losses += 1

        Console_UI().inform_user(f'Saved {no_networks} networks and {no_losses} losses to disk' +
                                 f' check out: {File_Manager().get_network_dir_path()}' +
                                 '\n*****************************************\n')

    def update_learning_rate(self, learning_rate):
        learning_rate = learning_rate * (self._experiment_set.batch_size / 128)
        for model_name in self.graph_travers_order:
            self.models[model_name].update_learning_rate(learning_rate)
        for loss_name in self.losses.keys():
            self.losses[loss_name].update_learning_rate(learning_rate)

    def reset_optimizers(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].reset_optimizers()

    def update_stochastic_weighted_average_parameters(self):
        has_run_average = False
        for model_name in self.graph_travers_order:
            model_has_run_average = self.models[model_name].update_stochastic_weighted_average_parameters()
            if model_has_run_average:
                has_run_average = True
        for loss_name in self.losses.keys():
            loss_has_run_average = self.losses[loss_name].update_stochastic_weighted_average_parameters()
            if loss_has_run_average:
                has_run_average = True

        return has_run_average

    def prepare_for_batchnorm_update(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].prepare_for_batchnorm_update()
        for loss_name in self.losses.keys():
            self.losses[loss_name].prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        self.encode(batch)
        if self.reconstruction:
            self.decode(batch)
        for model_name in self.graph_travers_order:
            self.models[model_name].update_batchnorm(batch)

        for loss_name in self.losses.keys():
            self.losses[loss_name](batch)
            self.losses[loss_name].update_batchnorm(batch)

    def finish_batchnorm_update(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].finish_batchnorm_update()
        for loss_name in self.losses.keys():
            self.losses[loss_name].finish_batchnorm_update()

    def __init_graph(self) -> nx.Graph:
        """Core graph initialization

        Connects all the edges and paths, cleans the paths and drops any unused sections

        Raises:
            AttributeError: Unexpected node structure

        Returns:
            Graph: The graph with all the connections
        """
        G = nx.DiGraph()
        for modality_name, data in self.get_modalities().items():
            try:
                modality_cfgs = Modality_Cfg_Parser.model_validate(data).root.model_dump()
            except ValidationError as e:
                raise ValueError(f'Failed to handle {data} for {modality_name}, got: {e}')

            modality_cfgs.update({'node_type': 'modality'})
            G.add_node(modality_name.lower(), **modality_cfgs)

        for model_name, model in self.get_models().items():
            model_cfgs = model
            if isinstance(model, BaseModel):
                model_cfgs = model.dict()
            model_cfgs.update({'node_type': 'model'})
            G.add_node(model_name.lower(), **model_cfgs)
            for h in model_cfgs['heads']:
                G.add_edge(h, model_name.lower())
            for t in model_cfgs['tails']:
                G.add_edge(model_name.lower(), t)

        assert nx.is_directed_acyclic_graph(G), f'Graph for task "{self.get_name()}" is not a DAG'

        # Clean up the tree of paths/nodes that lack any purpose
        # - edges without input/output modality when not in reconstruction where
        #   dead ends can contain style information
        # - drop tails that have no end-point, this is usually the "identification" loss
        #   not being activated.
        def is_inactive_output(x: str) -> bool:
            # Keep all nodes with child nodes
            if G.out_degree(x) > 0:
                return False

            # In reconstruction style nodes are to be retained
            try:
                if G.nodes[x].get('type', 'path?') == 'Style':
                    if self.reconstruction:
                        return False

                    return True

            except AttributeError as e:
                raise e(f'Issue with node {x}: {e}')

            return G.nodes[x].get('modality') not in ('input', 'output')

        while len(nodes_2_drop := [x for x in G.nodes() if is_inactive_output(x)]) > 0:
            for node in nodes_2_drop:
                G.remove_node(node)
                self._experiment_set.drop_modality(name=node)

        nodes_with_dropped_tail = [
            x for x in G.nodes()
            if G.nodes[x].get('tails') and any([v not in list(G.nodes()) for v in G.nodes[x].get('tails')])
        ]
        for x in nodes_with_dropped_tail:
            G.nodes[x]['tails'] = [tail for tail in G.nodes[x]['tails'] if tail in G.nodes()]

        return G

    def __get_graph_traverse_order(self):
        ordered_nodes = list(nx.topological_sort(self.graph))
        try:
            ordered_models = [m for m in ordered_nodes if self.graph.nodes[m]['node_type'] == 'model']
            return ordered_models
        except KeyError as e:
            Console_UI().warn_user("You have probably missed a key with node_type:")
            Console_UI().warn_user([m for m in ordered_nodes if 'node_type' not in self.graph.nodes[m]])
            raise KeyError(f'Key not found: {e}')
        return None

    def __init_models_and_adjust_sizes(self):
        if self._cfgs.graph_settings is not None:
            assert all([k in self.graph_travers_order for k in self._cfgs.graph_settings.keys()]), \
                'All specific graph settings must have a corresponding graph defined'

        for model_name in self.graph_travers_order:
            model_dict_args = {
                **self.graph.nodes[model_name],
                'optimizer_type': self.optimizer_type,
            }
            if (self._cfgs.graph_settings is not None and model_name in self._cfgs.graph_settings):
                settings = self._cfgs.graph_settings[model_name].dict()
                for key, value in settings.items():
                    if value is not None:
                        model_dict_args['neural_net_cfgs'][key] = value

            cfg_object = Model_Cfg(**model_dict_args)
            self.models[model_name] = Model(
                model_name=model_name,
                model_cfgs=cfg_object,
                graph=self.graph,
                experiment_set=self._experiment_set,
            )

    def __init_explicit_modalities(self):
        for modality_name, modality_cfgs in self.get_modalities().items():
            if modality_cfgs['type'] != 'Implicit':
                self._experiment_set.get_modality(modality_name, modality_cfgs)

    def __init_remaining_modalities(self):
        ordered_nodes = list(nx.topological_sort(self.graph))
        ordered_modalities = [m for m in ordered_nodes if self.graph.nodes[m]['node_type'] == 'modality']
        for m in ordered_modalities:
            # The get_modality inits but there is no need to use the returned modality here
            self._experiment_set.get_modality(m, self.graph.nodes[m])
            self.graph.nodes[m].update(self._experiment_set.get_modality_cfgs(m))

    def __init_losses(self):
        for loss_name, loss_cfgs in self.__get_losses().items():
            loss_cfgs.update({'node_type': 'loss'})
            self.graph.add_node(loss_name, **loss_cfgs)
            self.graph.add_edge(loss_cfgs['modality_name'], loss_name)

        ordered_nodes = list(nx.topological_sort(self.graph))
        ordered_losses = [loss for loss in ordered_nodes if self.graph.nodes[loss]['node_type'] == 'loss']

        for loss_name in ordered_losses:
            loss_dict = {
                **self.graph.nodes[loss_name],
                'apply': self._cfgs.apply,
                'has_pseudo_labels': self._cfgs.has_pseudo_labels,
                'pseudo_loss_factor': self._cfgs.pseudo_loss_factor,
                'optimizer_type': self._cfgs.optimizer_type,
                'ignore_index': self._experiment_set.ignore_index,
                'initial_learning_rate': self._cfgs.learning_rate.starting_value,
                'view_pool': self._cfgs.view_pool,
                'jitter_pool': self._cfgs.jitter_pool,
                'min_channels': self._cfgs.min_channels,
            }
            try:
                loss_cfgs = Loss_Cfg_Parser.model_validate(loss_dict).root
            except ValidationError as e:
                raise ValueError(f'Failed to handle {loss_dict} for {loss_name}, got: {e}')

            loss_class = get_loss_type(loss_type=loss_cfgs.loss_type, graph=self)
            self.losses[loss_name] = loss_class(experiment_set=self._experiment_set,
                                                loss_name=loss_name,
                                                loss_cfgs=loss_cfgs)

    def __get_losses(self):
        return {
            **self.__get_classification_loss(),
            **self.__get_reconstruction_loss(),
            **self.__get_discriminator_loss(),
            **self.__get_identification_loss(),
            **self.__get_regression_loss()
        }

    def __get_classification_loss(self):
        classification_loss = {}
        if self.classification or self.pi_model:
            for modality_name in self.get_experiment_explicit_modalities():
                modality = self._experiment_set.get_modality(modality_name)
                if modality.has_classification_loss():
                    classification_loss[modality.get_classification_loss_name()] = \
                        modality.get_classification_loss_cfgs()
        return classification_loss

    def __get_regression_loss(self):
        reg_loss = {}
        if self.regression:
            for modality_name in self.get_experiment_explicit_modalities():
                modality = self._experiment_set.get_modality(modality_name)
                if modality.has_regression_loss():
                    reg_loss[modality.get_regression_loss_name()] = \
                        modality.get_regression_loss_cfgs()
        return reg_loss

    def __get_reconstruction_loss(self):
        reconstruction_loss = {}
        if self.reconstruction:
            for modality_name in self.get_modalities():
                modality = self._experiment_set.get_modality(modality_name)
                if modality.has_reconstruction_loss() and (modality.is_input_modality()
                                                           or modality.is_implicit_modality()):
                    reconstruction_loss[modality.get_reconstruction_loss_name()] = \
                        modality.get_reconstruction_loss_cfgs()
        return reconstruction_loss

    def __get_discriminator_loss(self):
        discriminator_loss = {}
        if self.real_fake:
            for modality_name in self.get_modalities():
                modality = self._experiment_set.get_modality(modality_name)
                if (modality.is_input_modality() and self.reconstruction) or\
                   modality.has_discriminator_loss():
                    discriminator_loss[modality.get_discriminator_loss_name()] = \
                        modality.get_discriminator_loss_cfgs()
        return discriminator_loss

    def __get_identification_loss(self):
        identification_loss = {}
        if self.identification:
            for modality_name in self.get_modalities():
                modality = self._experiment_set.get_modality(modality_name)
                if modality.has_identification_loss():
                    identification_loss[modality.get_identification_loss_name()] = \
                        modality.get_identification_loss_cfgs()

        return identification_loss

    def drop_model_networks(self):
        """
        In order to save space + make sure we always use the last saved model
        when evaluating it is safest to clear the encoders and decoders and
        reload them from the disk.

        TODO: This could be implemented using a Singleton that has all the models in a dict
        """
        for model_name in self.graph_travers_order:
            self.models[model_name].drop_networks()

    def reloadModelNetworks(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].drop_networks()
            self.models[model_name].get_encoder()
            if self.reconstruction:
                self.models[model_name].get_decoder()
