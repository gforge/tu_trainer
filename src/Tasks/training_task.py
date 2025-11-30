import math
import time
from typing import Any, Dict, Tuple
import torch
from collections import defaultdict
from DataTypes import TaskType

from UIs.console_UI import Console_UI
from UIs.scene_UI_manager import SceneUIManager
from Tasks.base_task import Base_Task
from GeneralHelpers import ProgressBar
from .memory_profiler import MemoryProfiler


class Training_Task(Base_Task, MemoryProfiler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.task_type == TaskType.training,\
            f'training task {self.name} created for a non-training scenario: {self.scenario_name}'

    def __len__(self) -> int:
        """The number of iterations for an epoch to run"""
        return math.ceil(len(self.dataset.experiments[self.train_set_name]) / self._cfgs.backprop_every)

    def step(
        self,
        iteration_counter: int,
        scene_name: str,
    ):
        self.train_graph.zero_grad()

        for _ in range(self._cfgs.backprop_every):
            # Due to the parallel nature of loading the data we need to reset
            # the start time for the batch in order to get the true processing
            start_time = time.time()

            batch, new_epoch = self.__retrieve_batch_2_train(iteration_counter=iteration_counter, scene_name=scene_name)

            # We have reset the epoch as we reached the end during retrieval
            # - note that due to that we run backprop in the end_epoch
            #   the number of runs before backprop will be less that
            #   the backprop_every specifies but this should have minimal impact
            if new_epoch:
                start_time = time.time()

            batch['time']['start'] = start_time
            success = self.train_graph.train(batch)

            # Run the backward to get the gradients and drop the
            # memory graph that takse a lot of space
            start_time = time.time()
            self.train_graph.backward()
            batch['time']['backward'][self.get_name()] = {'start': start_time, 'end': time.time()}

        # Update weights
        self.train_graph.step()

        self.train_graph.collect_runtime_stats(batch)

        if not success:
            return False

        Console_UI().output_training_results(batch)
        return True

    def __retrieve_batch_2_train(self, iteration_counter: int, scene_name: str) -> Tuple[Dict[str, Any], bool]:
        """Retreives a batch

        If epoch has ended it will also run the """
        train_experiment_set = self.dataset.experiments[self.train_set_name]
        new_epoch = False
        batch = next(train_experiment_set)
        if batch is None:
            self.end_epoch(experiment_name=self.train_set_name, scene_name=scene_name)

            # Validate if divisable or if early on - we want to evaluate more in te
            # beginning of the training to see that everything works as expected
            if ((self.epoch < 5 and iteration_counter < 1e3)
                    or self.epoch % self._cfgs.validate_when_epoch_is_devisable_by == 0):
                self.validate(iteration_counter, scene_name=scene_name)

            batch = next(train_experiment_set)
            new_epoch = True
            if batch is None:
                raise RuntimeError('The next batch after resetting was empty!?')

        batch.update({
            'epoch': self.epoch,
            'graph_name': self.train_graph.get_name(),
            'task_name': self.get_name(),
            'iteration_counter': iteration_counter,
        })

        return batch, new_epoch

    def validate(
        self,
        iteration_counter,
        scene_name: str,
        set_name=None,
    ):
        if set_name is None:
            set_name = self.val_set_name

        experiment_set = self._get_experiment(name=set_name)
        experiment_set.set_view_dropout(None)

        self.__flush_data()

        bar = ProgressBar(total=len(experiment_set))
        at = f'@ epoch no. {self.epoch} for {SceneUIManager().name}'
        Console_UI().inform_user(f'Evaluating {self} {at}: {set_name} for {bar.total} steps')
        last_hooks = None
        with torch.no_grad():
            for batch in experiment_set:
                if batch is None:
                    bar.done()
                    break

                bar.current += 1
                bar()

                batch.update({
                    'epoch': self.epoch,
                    'graph_name': self.graphs[set_name.lower()].get_name(),
                    'task_name': self.get_name(),
                    'iteration_counter': iteration_counter,
                })
                self.graphs[set_name].eval(batch)
                last_hooks = batch['results_hooks']

        assert last_hooks is not None, 'Hooks should always exist and be run before saving data'
        fake_batch = {'results': defaultdict(dict)}
        for hook in last_hooks:
            hook(batch=fake_batch)
        self.__flush_data()

        self.end_epoch(set_name, scene_name=scene_name)

    def test(self, iteration_counter, scene_name):
        if not self.test_set_name:
            return None

        return self.validate(iteration_counter=iteration_counter, scene_name=scene_name, set_name=self.test_set_name)

    def end_epoch(self, experiment_name: str, scene_name: str):
        self.__flush_data()

        experiment_set = self.dataset.experiments[experiment_name]
        summary = {'epoch': self.epoch, 'graph_name': self.graph_name, 'task_name': self.get_name()}

        if experiment_name == self.train_set_name:
            summary['epoch_size'] = len(experiment_set)
            self.save(scene_name='last')
            self.epoch += 1

        experiment_set.end_epoch(summary=summary, scene_name=scene_name)
        Console_UI().add_epoch_results(summary)

    def update_learning_rate(self, learning_rate):
        self.graphs[self.train_set_name].update_learning_rate(learning_rate)
        self.graphs[self.val_set_name].update_learning_rate(learning_rate)
        if self.test_set_name:
            self.graphs[self.test_set_name].update_learning_rate(learning_rate)

    def reset_optimizers(self):
        self.graphs[self.train_set_name].reset_optimizers()
        self.graphs[self.val_set_name].reset_optimizers()
        if self.test_set_name:
            self.graphs[self.test_set_name].reset_optimizers()

    def save(self, scene_name='last'):
        self.graphs[self.train_set_name].save(scene_name)

    def stochastic_weight_average(self):
        self.__flush_data()
        experiment_set = self.dataset.experiments[self.train_set_name]
        set_name = self.train_set_name.lower()
        try:
            has_run_average = self.graphs[set_name].update_stochastic_weighted_average_parameters()  # noqa: F841
        except Exception as e:
            print(f'Error on {set_name}')
            raise e

        # Think this through - we can probably skip this step but it doesn't harm anything
        # if not has_run_average:
        #     return False

        self.graphs[set_name].prepare_for_batchnorm_update()
        self.graphs[set_name].train()
        experiment_set.reset_epoch()

        with torch.no_grad():
            while True:
                last_hooks = None

                Console_UI().inform_user('==>updating batchnorm')
                i = 0
                for batch in experiment_set:
                    i += 1
                    if i % 100 == 1:
                        Console_UI().inform_user(
                            f'Updating batchnorm for {self}, doing {self.train_set_name} on step {i}')

                    if batch is None:
                        self.graphs[set_name].finish_batchnorm_update()
                        assert last_hooks is not None, 'Hooks should always exist and be run before saving data'
                        fake_batch = {'results': defaultdict(dict)}
                        for hook in last_hooks:
                            hook(batch=fake_batch)
                        self.__flush_data()

                        return

                    self.graphs[set_name].update_batchnorm(batch)
                    last_hooks = batch['results_hooks']

    def __flush_data(self):
        """Clean memory

        Tries to flush all data in memory so that nothing remains
        as we progress to te next step. This includes doing any
        backpropagation that is stuck mid-way.
        """
        self.train_graph.backward()
        self.train_graph.step()
        self.train_graph.zero_grad()
        Console_UI().flush_all_saved_data()
