from typing import Dict
import numpy as np

from DataTypes import Learning_Rate_Function, Task_Cfg_Extended, TaskType
from DataTypes.Scene.extended import Scene_Cfg_Extended
from global_cfgs import Global_Cfgs
from file_manager import File_Manager
from UIs.console_UI import Console_UI
from Tasks.training_task import Training_Task


class Scene:
    """ 
    Runs a certain setup
    
    An experiment consisting of a particular network, data set and outcome.

    Run length = epochs * repeats times
    """

    def __init__(self, cfgs: Scene_Cfg_Extended):
        self.__cfgs = cfgs

        self.__tasks: Dict[str, Training_Task] = {}
        for task_name, task_cfgs_raw in self.__cfgs.tasks.items():
            if task_cfgs_raw.task_type == TaskType.training:
                task_cfgs = Task_Cfg_Extended(**{
                    **task_cfgs_raw.dict(),
                    'scene_cfgs': self.__cfgs,
                    'name': task_name,
                })
                self.__tasks[task_name] = Training_Task(cfgs=task_cfgs)
            else:
                raise KeyError(f'Unknown task type {task_cfgs_raw.task_type}')

        self.__epoch_size = len(self.__tasks[self.__main_task])

        if Global_Cfgs().get('test_run'):
            self.__cfgs.epochs = 2
            if self.__cfgs.repeat > 3:
                self.__cfgs.repeat = 3

        self.__task_load_balancer = {}  # Keeps track of the number of runs per task

        self.__logged_memory_usage = not Global_Cfgs().get('check_model_size')

    @property
    def __stochastic_weight_averaging(self) -> bool:
        return self.__cfgs.stochastic_weight_averaging

    @property
    def __stochastic_weight_averaging_last(self) -> bool:
        return self.__cfgs.stochastic_weight_averaging_last

    @property
    def __main_task(self) -> str:
        return self.__cfgs.main_task

    @property
    def __epochs(self) -> int:
        return self.__cfgs.epochs

    @property
    def __repeat(self) -> int:
        return self.__cfgs.repeat

    def __len__(self):
        return self.__epochs * self.__repeat

    iteration_counter = 0

    def should_task_run(self, task_name, task):
        """
        If certain task are much smaller than the current task then we should
        skip that task a few times or small datasets risk overfitting
        """
        if task_name != self.__main_task and self.__epoch_size > len(task):
            if task_name in self.__task_load_balancer:
                self.__task_load_balancer[task_name] += 1
                if self.__task_load_balancer[task_name] * len(task) % self.__epoch_size > len(task):
                    return False
            else:
                self.__task_load_balancer[task_name] = 0
        return True

    def log_memory_usage(self):
        ui = Console_UI()
        for key in self.__tasks.keys():
            task = self.__tasks[key]
            memory_usage = task.get_memory_usage_profile()
            File_Manager().write_usage_profile(
                scene_name=self.__scene_name,
                task=key,
                memory_usage=memory_usage,
            )
            ui.inform_user(f'\n Memory usage for {self.get_name()}::{key}\n')
            ui.inform_user(memory_usage)
        self.__logged_memory_usage = True

    def update_learning_rate(self, epoch_no: int):
        for task in self.__tasks.values():
            task.update_learning_rate(self.__get_learning_rate(epoch_no))

    def reset_optimizers(self):
        for task in self.__tasks.values():
            task.reset_optimizers()

    def run_epoch(self, epoch_no):
        self.update_learning_rate(epoch_no=epoch_no)

        for _ in range(self.__epoch_size):
            for key, task in self.__tasks.items():
                if self.should_task_run(task_name=key, task=task):
                    task.step(iteration_counter=Scene.iteration_counter, scene_name=self.get_name())
            Scene.iteration_counter += 1

            if self.__logged_memory_usage is False:
                self.log_memory_usage()

        for task in self.__tasks.values():
            task.save(scene_name='last')
            # Not really helping with just emptying cache - we need to add something more
            # removing as this may be the cause for errors
            # torch.cuda.empty_cache()

    def run_scene(self, start_epoch=0):
        ui = Console_UI()
        ui.overall_total_epochs = self.__epochs
        ui.overall_total_repeats = self.__repeat

        Global_Cfgs().set_forward_noise(self.__cfgs.forward_noise)
        Global_Cfgs().set_use_custom_dropout(self.__cfgs.use_custom_dropout)
        self.reset_optimizers()

        for r in range(self.__repeat):
            ui.overall_repeat = r
            if (self.__stochastic_weight_averaging and r > 0):
                self.__tasks[self.__main_task].stochastic_weight_average()

            for epoch_no in range(self.__epochs):
                try:
                    ui.overall_epoch = epoch_no
                    if start_epoch > epoch_no + r * self.__epochs:
                        Scene.iteration_counter += self.__epoch_size
                    else:
                        self.run_epoch(epoch_no=epoch_no)
                except KeyboardInterrupt:
                    input("\nPress ctrl+c again to quit")

        ui.reset_overall()

        # Note that the evaluation happens after this step and therefore averaging may hurt the performance
        if self.__stochastic_weight_averaging_last:
            self.__tasks[self.__main_task].stochastic_weight_average()
            for task in self.__tasks.values():
                task.save(scene_name='last')

        self.run_evaluation_and_test()

        # Save all tasks before enterering the next scene
        for task in self.__tasks.values():
            task.save(scene_name=self.get_name())
            [g.drop_model_networks() for g in task.graphs.values()]
            # Not really helping with just emptying cache - we need to add something more
            # removing as this may be the cause for errors
            # torch.cuda.empty_cache()

    def get_name(self):
        return self.__cfgs.name

    def run_evaluation_and_test(self, scene_name=None):
        if scene_name is None:
            scene_name = self.get_name()

        for task in self.__tasks.values():
            task.validate(iteration_counter=Scene.iteration_counter, scene_name=scene_name)
            task.test(iteration_counter=Scene.iteration_counter, scene_name=scene_name)

    def __get_learning_rate(self, epoch: int) -> np.ndarray:
        """Converts learning rate to a containing the learning rate for an epoch

        Args:
            epoch (int): The epoch that we want to retrieve

        Raises:
            ValueError: If a learning rate function isn't implemented

        Returns:
            np.ndarray: [description]
        """
        if self.__cfgs.learning_rate.function == Learning_Rate_Function.cosine:
            lsp = np.linspace(0, np.pi / 2, self.__epochs + 1, dtype='float32')
            return np.cos(lsp)[epoch] * self.__cfgs.learning_rate.starting_value

        if self.__cfgs.learning_rate.function == Learning_Rate_Function.linear:
            lsp = np.linspace(1, 0, self.__epochs + 1, dtype='float32')
            return lsp[epoch] * self.__cfgs.learning_rate.starting_value

        raise ValueError(f'Learning rate function {self.__cfgs.learning_rate.function}')
