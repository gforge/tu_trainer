import os
from abc import ABCMeta
from typing import Any, Dict, Optional

from networkx import DiGraph
from pydantic.main import BaseModel

from DataTypes import Graph_Cfg_Extended
from DataTypes.enums import OptimizerType
from global_cfgs import Global_Cfgs
from UIs.console_UI import Console_UI
from Datasets.experiment_set import Experiment_Set


class Base_Graph(metaclass=ABCMeta):
    """ 
    Abstratct base class for Graph.

    **Graph** is an abstract object that maps the relationship between the
    input and output modalities. The most simple form of graph could be
    just a deep convolution network mapping the input to the output.

    The base graph is mostly for handling access to the configuration
    """

    def __init__(
        self,
        experiment_set: Experiment_Set,
        cfgs: Graph_Cfg_Extended,
    ):
        # The graph is set a little later and thus we
        # will initially have a None in te graph variable
        self.graph: Optional[DiGraph] = None
        self._experiment_set = experiment_set
        self.experiment_name = self._experiment_set.get_name()
        self._cfgs = cfgs

        self.__inform_user_of_graph_structure()

    @property
    def name(self) -> str:
        return self._cfgs.name

    @property
    def classification(self) -> bool:
        return self._cfgs.apply.classification

    @property
    def reconstruction(self) -> bool:
        return self._cfgs.apply.reconstruction

    @property
    def identification(self) -> bool:
        return self._cfgs.apply.identification

    @property
    def regression(self) -> bool:
        return self._cfgs.apply.regression

    @property
    def pi_model(self) -> bool:
        return self._cfgs.apply.pi_model

    @property
    def real_fake(self) -> bool:
        return self._cfgs.apply.real_fake

    @property
    def optimizer_type(self) -> OptimizerType:
        return self._cfgs.optimizer_type

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all modalities active wihtin the graph

        Filters modalities on the active graph once this has been created

        Returns:
            Dict[str, Any]: A dictionary with key as the name and modality configs as value
        """
        all_modalities = {**self.get_graph_specific_modalities(), **self.get_experiment_modalities()}
        # Filter modalities based on the graph
        if self.graph is not None:
            all_modalities = {k: v for k, v in all_modalities.items() if k in self.graph.nodes()}

        modality_dict: Dict[str, Dict[str, Any]] = {
            k: v.dict() if isinstance(v, BaseModel) else v for k, v in all_modalities.items()
        }

        return modality_dict

    def get_explicit_modalities(self):
        return {**self.get_experiment_explicit_modalities(), **self.get_graph_specific_explicit_modalities()}

    def get_experiment_modalities(self):
        return {**self.get_experiment_explicit_modalities(), **self.get_experiment_implicit_modalities()}

    def get_experiment_explicit_modalities(self):
        modalities_cfgs = {}
        explicit_modality_names = self._cfgs.modalities.experiment_modalities
        if not self.identification:
            explicit_modality_names = [n for n in explicit_modality_names if n != 'id']

        for modality_name in explicit_modality_names:
            modalities_cfgs[modality_name] = self._experiment_set.get_modality_cfgs(modality_name)
        return modalities_cfgs

    def get_experiment_implicit_modalities(self):
        implicit_modalities_cfgs = {}
        for modality_name, modality_cfgs in self.get_experiment_explicit_modalities().items():
            es = self._experiment_set
            explicit_modality = es.get_modality(modality_name, modality_cfgs)
            implicit_modality_name = es.get_implicit_modality_name(explicit_modality.get_name())
            implicit_modalities_cfgs[implicit_modality_name] = es.get_modality_cfgs(implicit_modality_name)
        return implicit_modalities_cfgs

    def get_graph_specific_modalities(self):
        return {**self.get_graph_specific_explicit_modalities(), **self.get_graph_specific_implicit_modalities()}

    def get_pseudo_explicit_modalities(self):
        modalities_cfgs = {}
        return modalities_cfgs

    def get_pseudo_implicit_modalities(self):
        modalities_cfgs = {}
        return modalities_cfgs

    def get_graph_specific_explicit_modalities(self):
        modalities = self._cfgs.modalities.graph_specific_modalities
        return {name: cfgs for name, cfgs in modalities.items() if (cfgs.type != 'Implicit')}

    def get_graph_specific_implicit_modalities(self):
        modalities = self._cfgs.modalities.graph_specific_modalities
        implicit_modalities_cfgs = {name: cfgs for name, cfgs in modalities.items() if (cfgs.type == 'Implicit')}

        for modality_name, modality_cfgs in\
                self.get_graph_specific_explicit_modalities().items():
            explicit_modality = \
                self._experiment_set.get_modality(modality_name, modality_cfgs)
            implicit_modalities_cfgs[explicit_modality.get_implicit_modality_name()] = \
                explicit_modality.get_implicit_modality_cfgs()
        return implicit_modalities_cfgs

    def get_models(self):
        return {**self.get_explicit_models(), **self.get_implicit_models()}

    def get_implicit_models(self):
        models_cfgs = {}
        explicit_modalities = self.get_explicit_modalities()
        for modality_name, _ in explicit_modalities.items():
            models_cfgs[self._experiment_set.get_model_name(modality_name)] =\
                self._experiment_set.get_model_cfgs(modality_name)
        return models_cfgs

    def get_explicit_models(self):
        return self._cfgs.models.graph_specific_models

    def get_name(self) -> str:
        return self.name

    def _save_graph_to_mermaid_file(self) -> None:
        """Save mermaidjs description of graph to log folder
        """
        if not os.path.exists(Global_Cfgs().log_folder):
            os.makedirs(Global_Cfgs().log_folder, exist_ok=True)
        fn = 'mermaidjs_{ds_name}_{graph_name}_{exp_name}_{scene_name}.txt'.format(
            ds_name=self._cfgs.dataset_name,
            graph_name=self.get_name(),
            exp_name=self.experiment_name,
            scene_name=self._cfgs.scene_name,
        )
        mermain_fn = os.path.join(Global_Cfgs().log_folder, fn)
        with open(mermain_fn, 'w') as mermain_file:
            mermain_file.write(self.__convert_to_mermaidjs())
        Console_UI().inform_user(f'Wrote mermaidjs config to {mermain_fn}')

    def _convert_to_mermaidjs_graph(self):
        chart = 'graph LR\n'
        for (head, tail) in list(self.graph.edges):
            if self.graph.nodes[tail]['node_type'] == 'modality':
                head_id = head.replace(' ', '_')
                tail_shape_str = self._experiment_set.get_modality(tail).get_shape_str()
                tail_id = tail.replace(' ', '_')
                tail_txt = tail.replace('_', '<br/>')
                chart += f'\t{head_id} --> |{tail_shape_str}|{tail_id}(({tail_txt}))\n'

            elif self.graph.nodes[tail]['node_type'] == 'loss':
                # Generates a box around the loss
                tail_id = tail.replace(' ', '_')
                modality_name = self.graph.nodes[tail]['modality_name'].replace(' ', '_')
                chart += f'\tsubgraph {tail_id}\n\t\t{modality_name}\n\tend\n'

            else:
                head_id = head.replace(' ', '_')
                head_shape_str = self._experiment_set.get_modality(head).get_shape_str()
                tail_id = tail.replace(' ', '_')
                tail_txt = tail.replace('_', '<br/>')
                chart += f'\t{head_id} --> |{head_shape_str}|{tail_id}[{tail_txt}]\n'
        return chart.replace('style', 'Style')

    def _convert_to_mermaidjs_gantt(self) -> str:
        gantt = 'gantt\n'
        if self.time_frame is not None:
            x = self.time_frame
            x.sort_values('start', inplace=True)

            gantt += 'title A cycle time in ms\n'
            prev_section = ''
            for i in range(len(x)):
                t = x.iloc[i]
                section, name = t.name
                if section != prev_section:
                    gantt += '\tsection %s\n' % section
                    prev_section = section
                gantt += '\t\t%s: %04d, %04d\n' % (name, 1000 * t['start'], 1000 * t['end'])
        return gantt.replace('style', 'Style')

    def __convert_to_mermaidjs(self, to_visualize='graph') -> str:
        if to_visualize.lower() == 'graph'.lower():
            return self._convert_to_mermaidjs_graph()

        return self._convert_to_mermaidjs_gantt()

    def __inform_user_of_graph_structure(self):
        if not Global_Cfgs().get('silent_init_info') and Global_Cfgs().get('graph_info'):
            UI = Console_UI()
            UI.inform_user(
                info=['explicit experiment modalities',
                      list(self.get_experiment_explicit_modalities().keys())],
                debug=self.get_experiment_explicit_modalities(),
            )
            UI.inform_user(
                info=['implicit experiment modalities',
                      list(self.get_experiment_implicit_modalities().keys())],
                debug=self.get_experiment_implicit_modalities(),
            )
            UI.inform_user(
                info=['explicit graph modalities',
                      list(self.get_graph_specific_explicit_modalities().keys())],
                debug=self.get_graph_specific_explicit_modalities(),
            )
            UI.inform_user(
                info=['implicit graph modalities',
                      list(self.get_graph_specific_implicit_modalities().keys())],
                debug=self.get_graph_specific_implicit_modalities(),
            )
            UI.inform_user(
                info=['explicit models', list(self.get_explicit_models().keys())],
                debug=self.get_explicit_models(),
            )
            UI.inform_user(
                info=['implicit models', list(self.get_implicit_models().keys())],
                debug=self.get_implicit_models(),
            )
