import time
from DataTypes import Purpose
from networkx import Graph
from DataTypes import Model_Cfg, Network_Cfg_Cascade_Raw, Network_Fork_Cfg_Raw, \
    Network_Cfg_Fully_Connected_Raw, Network_Cfg_Morph_Raw, Network_Cfg_Pre_Defined_Raw

from Graphs.Factories.cascade_factory import Cascade_Factory
from Graphs.Factories.pre_defined_factory import Pre_Defined_Factory
from Graphs.Factories.fully_connected_factory import Fully_Connected_Factory
from Graphs.Factories.fork_factory import Fork_Factory
from Graphs.Factories.morph_factory import Morph_Factory


class Model:
    """
    **Model** is an object that maps two **modalities** together.

    A **Model** must have the following objects:

    1. `encoder` and `decoder` are the `neural_nets` that does the mapping
    2. `heads` and `tails` are the head and tail **modalities**
    3. `factory` is a factory that create singleton `encoder` and `decoder`.
        The reason that these neural networks are singelton is that we want
        to make sure that different graphs and tasks share the same mappings
        between modalities. (for example, during the train and test, we should
        have the same neural networks do the mappings)

    **Model** also has the following functions:

    1. `__init__` initializes the model based on `model_cfgs`, `graph_cfgs`,
        `scene_cfgs` and `scenario_cfgs`.
    2. `encode(batch)` and `decode(batch)` forwards the `batch` into `encoder`
        and `decoder` neural networks, which in turn, calculate the results and
        put it back in the `batch`. `batch` is a dictionary that stores all the
        modality `tensors`, `loss` values, `results` and `times` in it.
    3. `update_modality_dims()` is an important function that automatically
        calculate the dimensions of the modalities if they are not specified in
        other config files. The reason that we calculate the modality dimension
        on the fly is that this way, we can dynamically make the graph and don't
        be bothered with the input,output sizes of tensors.
    4. `step()`, `zero_grad()`, `update_learning_rate(learning_rate)`, `train()`
        and `eval()` are calling the same functionalities in the `encoder` and
        `decoder` neural networks, if they exists.
    """

    def __init__(
        self,
        graph: Graph,
        experiment_set,
        model_name: str,
        model_cfgs: Model_Cfg,
    ):
        self.__graph = graph
        self.__model_name = model_name
        self.__cfgs = model_cfgs

        self.experiment_set = experiment_set
        self.__encoder = None
        self.__decoder = None
        self.__factory = self.get_factory()

        self.__update_modality_dims()

    def encode(self, batch):
        start_time = time.time()
        encoder = self.__get_encoder()
        encoder(batch)
        batch['time']['encode'][self.get_name()] = {'start': start_time, 'end': time.time()}

    def __get_encoder(self):
        if self.__encoder is None:
            self.__encoder = self.__init_neural_net(purpose=Purpose.encoder)
        return self.__encoder

    def decode(self, batch):
        start_time = time.time()
        decoder = self.__get_decoder()
        decoder(batch)
        batch['time']['decode'][self.get_name()] = {'start': start_time, 'end': time.time()}

    def __get_decoder(self):
        if self.__decoder is None:
            self.__decoder = self.__init_neural_net(purpose=Purpose.decoder)
        return self.__decoder

    def drop_networks(self):
        self.__decoder = None
        self.__encoder = None

    def __init_neural_net(self, purpose: Purpose):
        neural_net = self.__factory.get_neural_net(model_cfgs=self.__cfgs,
                                                   heads={h: self.__graph.nodes[h] for h in self.__cfgs.heads},
                                                   tails={t: self.__graph.nodes[t] for t in self.__cfgs.tails},
                                                   purpose=purpose,
                                                   optimizer_type=self.__cfgs.optimizer_type)

        neural_net.update_learning_rate(self.initial_learning_rate)
        return neural_net

    def get_factory(self):
        neural_net_cfgs = self.__cfgs.neural_net_cfgs
        if isinstance(neural_net_cfgs, Network_Cfg_Cascade_Raw):
            return Cascade_Factory()

        if isinstance(neural_net_cfgs, Network_Cfg_Pre_Defined_Raw):
            return Pre_Defined_Factory()

        if isinstance(neural_net_cfgs, Network_Cfg_Fully_Connected_Raw):
            return Fully_Connected_Factory()

        if isinstance(neural_net_cfgs, Network_Fork_Cfg_Raw):
            return Fork_Factory()

        if isinstance(neural_net_cfgs, Network_Cfg_Morph_Raw):
            return Morph_Factory()

        raise KeyError(f'Unsupported neural network definition "{type(neural_net_cfgs)}"')

    def zero_grad(self):
        try:
            if self.__encoder is not None:
                self.__encoder.zero_grad()
            if self.__decoder is not None:
                self.__decoder.zero_grad()
        except Exception as e:
            raise RuntimeError(f'Failed zero_grad for {self.get_name()}: {e}')

    def step(self):
        if self.__encoder is not None:
            self.__encoder.step()
        if self.__decoder is not None:
            self.__decoder.step()

    def update_learning_rate(self, learning_rate):
        if self.__encoder is not None:
            self.__encoder.update_learning_rate(learning_rate)
        else:
            self.initial_learning_rate = learning_rate
        if self.__decoder is not None:
            self.__decoder.update_learning_rate(learning_rate)
        else:
            self.initial_learning_rate = learning_rate

    def reset_optimizers(self):
        if self.__encoder is not None:
            self.__encoder.reset_optimizer()

        if self.__decoder is not None:
            self.__decoder.reset_optimizer()

    def update_stochastic_weighted_average_parameters(self):
        has_run_average = False
        if self.__encoder is not None:
            encoder_has_run_average = self.__encoder.update_stochastic_weighted_average_parameters()
            if encoder_has_run_average:
                has_run_average = True
        if self.__decoder is not None:
            decoder_has_run_average = self.__decoder.update_stochastic_weighted_average_parameters()
            if decoder_has_run_average:
                has_run_average = True

        return has_run_average

    def prepare_for_batchnorm_update(self):
        if self.__encoder is not None:
            self.__encoder.prepare_for_batchnorm_update()
        if self.__decoder is not None:
            self.__decoder.prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        if self.__encoder is not None:
            self.__encoder.update_batchnorm(batch)
        if self.__decoder is not None:
            self.__decoder.update_batchnorm(batch)

    def finish_batchnorm_update(self):
        if self.__encoder is not None:
            self.__encoder.finish_batchnorm_update()
        if self.__decoder is not None:
            self.__decoder.finish_batchnorm_update()

    def get_name(self):
        return self.__model_name

    def train(self):
        """
        Activate train mode
        """
        if self.__encoder is not None:
            self.__encoder.train()
        if self.__decoder is not None:
            self.__decoder.train()

    def eval(self):
        if self.__encoder is not None:
            self.__encoder.eval()
        if self.__decoder is not None:
            self.__decoder.eval()

    def save(self, scene_name):
        if self.__encoder is not None:
            self.__encoder.save(scene_name)
        if self.__decoder is not None:
            self.__decoder.save(scene_name)

    def __update_modality_dims(self):
        self.__factory.update_modality_dims(neural_net_cfgs=self.__cfgs.neural_net_cfgs,
                                            heads=self.__cfgs.heads,
                                            tails=self.__cfgs.tails,
                                            graph=self.__graph)
