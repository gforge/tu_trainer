from collections import defaultdict
import getpass
from time import time
import pandas as pd
from datetime import datetime
from yaml import dump as yamlDump
from GeneralHelpers import Singleton
from typing import Dict, Any, Optional, TypeVar

from .scene_UI_manager import SceneUIManager
from .Writers import AllWriters

pd.options.display.max_rows = 999
pd.options.display.float_format = '{:,.02f}'.format

PrettyPrint = TypeVar('PrettyPrint', str, dict, list, pd.DataFrame, Any)


def _pretty_print(x: PrettyPrint, indent=0, force_print_dataframe=False):
    """
    A pretty printer
    """
    if isinstance(x, str):
        indentation = ''.join(['\t' for i in range(indent)])
        print(f'{indentation}{x}')
    elif isinstance(x, dict):
        print(yamlDump(x, width=200))
    elif isinstance(x, list):
        [_pretty_print(i, indent=indent + 1) for i in x]
    elif isinstance(x, pd.DataFrame):
        if not force_print_dataframe and (len(x.columns) > 10 or len(x.index) > 100):
            print(f'\nSkipped print of dataframe due to large size: {x.shape}')
        elif len(x.index) > 0:
            print(x.fillna(''))
        else:
            print('Empty dataframe?')
    else:
        print(x)


class Console_UI(metaclass=Singleton):
    pd.options.display.float_format = '{:,.02f}'.format

    def __init__(self, log_level: str, global_cfgs):
        log_levels = {"off": 0, "warning": 1, "info": 2, "debug": 3}
        self.log_level = log_levels[log_level]
        self.writer = AllWriters()
        self.scene_manager = SceneUIManager()
        self.global_cfgs = global_cfgs
        self.iteration = 0
        self.__last_hooks = {}
        self.__last_report_time = {}
        self.reset_overall()

    def reset_overall(self):
        self.flush_all_saved_data()
        self.overall_epoch = None
        self.overall_repeat = None
        self.overall_total_epochs = None
        self.overall_total_repeats = None
        self.__first_run = True

    def warn_user(self,
                  message: Optional[str] = None,
                  info: Optional[PrettyPrint] = None,
                  debug: Optional[PrettyPrint] = None):
        if self.log_level >= 1 and message is not None:
            _pretty_print(message, force_print_dataframe=True)
        self.inform_user(info, debug)

    def inform_user(self, info: Optional[PrettyPrint] = None, debug: Optional[PrettyPrint] = None):
        if self.log_level >= 2 and info is not None:
            _pretty_print(info)
        self.debug(debug)

    def debug(self, debug: Optional[PrettyPrint] = None):
        if self.log_level >= 3 and debug is not None:
            _pretty_print(debug)

    def receive_regular_input(self, message: str = 'Input:[Y/n]', default: str = 'Y'):
        return input(message) or default

    def receive_password(self, message: str = 'Please enter password'):
        return getpass.getpass(message)

    def warn_user_and_ask_to_continue(self, message: str = 'Warning!!!'):
        self.warn_user(message)
        answer = self.receive_regular_input('continue? [y/N]', default='n')
        if answer.lower().startswith('n'):
            exit()

    def inform_user_and_ask_to_continue(self, message: str = 'Info!!'):
        self.inform_user(message)
        answer = self.receive_regular_input('continue? [Y/n]', default='y')
        if answer.lower().startswith('n'):
            exit()

    def add_epoch_results(self, summary: dict):
        s = pd.DataFrame(summary['modalities']).T
        columns2change = [c[len('pseudo_'):] for c in s.columns if c.startswith('pseudo_')]
        rename_cols = {c: f'teacher_{c}' for c in columns2change}
        s.rename(columns=rename_cols, inplace=True)

        result_id = self.scene_manager.get_result_id(ds=summary["dataset_name"],
                                                     exp=summary['experiment_name'],
                                                     task=summary['task_name'],
                                                     graph=summary['graph_name'])

        iteration = self.iteration
        if 'epoch_size' in summary:
            iteration -= summary['epoch_size'] // 2
        self.writer.scalar_outcome.add_last_data(result_id=result_id, data=s, iteration=iteration)

        self.inform_user(s)
        self.writer.flush_all_results_2_tensorboard()

    def __get_scenario_info(self):
        info = f'Scenario: {self.global_cfgs.get("scenario")} >> {self.global_cfgs.start_time}'

        resume = self.global_cfgs.get("resume", False)
        if resume:
            info += f' [resumed from {resume}::{self.global_cfgs.get("resume_scene")}]'

        return info

    def __get_scene_info(self, batch: Dict[str, Any]):
        graph = batch['graph_name']
        task = batch['task_name']

        info = f'Scene: {self.scene_manager.name}, graph: {graph}, task: {task}'
        if self.overall_epoch is not None and self.overall_repeat is not None:
            epoch_info = f'epoch: {self.overall_epoch} (of {self.overall_total_epochs})'
            repeat_info = f'repeat: {self.overall_repeat} (of {self.overall_total_repeats})'
            info += f' {epoch_info} {repeat_info}'

        return info

    def __get_counter_info(self, batch: Dict[str, Any]):
        epoch = batch['epoch']
        index = batch['batch_index']
        size = batch['epoch_size']
        counter = batch['iteration_counter']
        batch_size = batch['batch_size_info']
        return f'Batch specifics: epoch[{epoch}][{index}/{size}], iteration[{counter}], batch_size: {batch_size}'

    def __get_time_info(self, batch: Dict[str, Any]):
        time_spent = batch['time_stats']['end'].groupby(level=0).max() -\
                     batch['time_stats']['start'].groupby(level=0).min()

        # When the software hangs it is nice to have some time info
        at_time = f'on {datetime.now():%Y-%m-%d %H:%M:%S}'
        total_ms = float(batch['time']['true_full_time']) * 1000
        batch_times = [f'{k}: {t*1000:4.0f}ms' for k, t in time_spent.to_dict().items()]

        time_spent_string = f'Time spent per batch: {total_ms:.0f}ms ({", ".join(batch_times)}) {at_time}'

        return time_spent_string

    def flush_all_saved_data(self, task_name: Optional[str] = None):
        """
        As data is only saved once the hooks run we must perform this as we are finishing each
        task.
        """

        fake_batch = {'results': defaultdict(dict)}

        if task_name is None:
            print('\n')
            # Due to the pop we need to have this new list
            all_task_names = [*self.__last_hooks.keys()]
            for tn in all_task_names:
                self.flush_all_saved_data(task_name=tn)
            # Should be enough with the pop() but just to make sure that everything is gone
            self.__last_hooks = {}

        elif task_name in self.__last_hooks:
            hooks = self.__last_hooks.pop(task_name)
            for hook in hooks:
                hook(batch=fake_batch)

            self.inform_user(f'Flushed {task_name} hooks')

    def __get_avg_time_report(self, task_name: str):
        time_since_last_report_msg = ''
        if task_name in self.__last_report_time:
            total_time_spent = time() - self.__last_report_time[task_name]['start']
            no_iterations = self.iteration - self.__last_report_time[task_name]['iteration']
            avg_time_spent = total_time_spent / no_iterations
            time_since_last_report_msg = f' (avg. time between batch {avg_time_spent:0.1f}s)'

        self.__last_report_time[task_name] = {'start': time(), 'iteration': self.iteration}

        return time_since_last_report_msg

    def __skip_report(self) -> bool:
        if self.__first_run:
            self.__first_run = False
            return False

        if self.iteration <= 5:
            return False

        if self.iteration < 50:
            if self.iteration % 10 == 0:
                return False
            else:
                return True

        if self.iteration % 50 == 0:
            return False

        return True

    def __run_hooks(self, batch: Dict[str, Any], task_name: str):
        if 'results_hooks' not in batch:
            return batch

        for hook in batch['results_hooks']:
            hook(batch=batch)

        del batch['results_hooks']

        if task_name in self.__last_hooks:
            del self.__last_hooks[task_name]

        return batch

    def output_training_results(self, batch: Dict[str, Any]):
        self.iteration = batch['iteration_counter']
        task_name = batch['task_name']

        if self.__skip_report():
            print('.', end='', flush=True)
            self.__last_hooks[task_name] = batch['results_hooks']
            return

        # Log the time that this takes

        report_time = time()
        self.__run_hooks(batch=batch, task_name=task_name)

        for _, v in batch['results'].items():
            nondisplay_items = [
                'output', 'pseudo_output', 'target', 'euclidean_distance', *self.writer.special_results_names
            ]
            for item in nondisplay_items:
                if item in v:
                    v.pop(item)

        s = pd.DataFrame(batch['results']).T
        self.inform_user(s)
        self.inform_user('-----------------')
        self.debug(batch['time_stats'])
        self.debug('-----------------')

        info_string = f'{self.__get_scenario_info()}\n{self.__get_scene_info(batch)}\n{self.__get_counter_info(batch)}'

        results_string = 'Batch summary: '
        results_string += ', '.join(['mean %s: %.2e' % (k, s[k].mean()) for k in s.keys()])

        report_time_msg = f'Time spent time reporting batch: {(time() - report_time)*1e3:0.0f}ms' + \
            self.__get_avg_time_report(task_name=task_name)
        self.inform_user(f'{info_string}\n{results_string}\n{self.__get_time_info(batch)}\n{report_time_msg}\n=======')

        # The runtime becomes huge and lacks anything really interesting...
        # self.add_scalar_to_tensorboard(s, result_name, self.iteration)
