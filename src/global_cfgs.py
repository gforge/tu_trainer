import os
from pathlib import Path
from posixpath import basename
import sys
import argparse
import subprocess
import re

from datetime import datetime
import pytz
from glob import iglob
import shutil
from UIs.console_UI import Console_UI
from Utils.retrieve_dir import retrieve_dir
from GeneralHelpers import Singleton
from file_manager import File_Manager
"""
This file stores variables that are constant during the runtime but will change
from runtime to tuntime
"""


def positive_float(string):
    value = float(string)
    if value < 0:
        msg = "%r is less than 0" % string
        raise argparse.ArgumentTypeError(msg)
    return value


class Global_Cfgs(metaclass=Singleton):
    """ This class parses the settings
    
    It parses the CLI arguments and also picks up main environment variables
    """

    def __init__(self, test_mode=False, test_init_values={}):
        if test_mode:
            self.cfgs = test_init_values

            Console_UI(self.get('log_level', 'warning'), global_cfgs=self)
            self.sub_log_path = self.get('sub_log_path', 'sub_log_not_set')

            File_Manager(annotations_root=self.get('annotation_root'),
                         log_folder=self.get('log_folder'),
                         scenario_log_root=self.get('scenario_log_root'),
                         resume_prefix=self.get('resume'),
                         resume_scene=self.get('resume_scene'),
                         tmp_root=self.get('tmp_root'),
                         model_zoo_root=self.get('model_zoo_root'),
                         global_cfgs=self)
            return

        args = self.parse_argument()
        if args.activate_torch_cuda_benchmark:
            import torch
            torch.backends.cudnn.benchmark = True

        self.cfgs = args.__dict__

        Console_UI(self.get('log_level', 'info'), global_cfgs=self)
        self.read_environment_variables()

        self.start_time = datetime.now(pytz.timezone('Europe/Stockholm')).strftime('%Y%m%d/%H.%M')
        run_id = self.start_time
        if self.get('test_run'):
            run_id = f'test_run_{run_id}'
        self.sub_log_path = os.path.join(self.get('scenario'), run_id)

        if self.get('resume') is not None:
            self.prep_resume()

        fm = File_Manager(scenario_log_root=self.scenario_log_root,
                          log_folder=self.log_folder,
                          annotations_root=self.get('annotation_root'),
                          resume_prefix=self.get('resume'),
                          resume_scene=self.get('resume_scene'),
                          tmp_root=self.get('tmp_root'),
                          model_zoo_root=self.get('model_zoo_root'),
                          global_cfgs=self)

        setup_data = {
            'call': ' '.join(sys.argv),
            'git': self.__git_details(),
            'run_id': run_id,
            'python_version': sys.version,
            'setup': self.cfgs,
        }

        fm.log_setup(data=setup_data, name=Path('base_setup.yaml'))

        self.__forward_noise = 0
        self.__use_custom_dropout = False

    def __git_details(self) -> str:
        try:
            branch = subprocess.check_output(["git", "symbolic-ref", "HEAD"], universal_newlines=True).strip()
            branch = re.sub(pattern="^refs/heads/", repl="", string=branch)
        except subprocess.CalledProcessError:
            branch = "?detached?"
        hash_id = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%H'],
                                          universal_newlines=True).strip()
        return f'{branch} ({hash_id})'

    @property
    def log_folder(self):
        return os.path.join(self.get('log_root'), self.sub_log_path)

    @property
    def scenario_log_root(self):
        return os.path.join(self.get('log_root'), self.get('scenario'))

    def set_forward_noise(self, forward_noise=1e-1):
        self.__forward_noise = forward_noise

    @property
    def forward_noise(self):
        return self.__forward_noise

    def set_use_custom_dropout(self, use_custom_dropout: bool):
        self.__use_custom_dropout = use_custom_dropout

    @property
    def use_custom_dropout(self):
        return self.__use_custom_dropout

    def parse_argument(self):
        parser = argparse.ArgumentParser("DeepMed")

        parser.add_argument(
            '--log_level',
            dest='log_level',
            default='Info',
            type=str.lower,
            choices=['off', 'warning', 'info', 'debug'],
            help='Different levels of logging',
        )

        parser.add_argument(
            '-scenario',
            dest='scenario',
            default='all_xray',
            type=str.lower,
            help='Scenario to run',
        )

        parser.add_argument(
            '-resume',
            dest='resume',
            default=None,
            type=str.lower,
            help='Resume prefix path. It should be in the form of YYYYMMDD/HH.MM',
        )

        parser.add_argument(
            '-rl',
            '--resume_last',
            dest='resume',
            action='store_const',
            const='last',
            help='find the last run and resume from there',
        )

        parser.add_argument(
            '-rs',
            '--resume_scene',
            dest='resume_scene',
            default='last',
            type=str,
            help='Resume a specific scene - defaults to last',
        )

        parser.add_argument(
            '-rc',
            '--resume_config',
            dest='resume_config',
            action='store_true',
            help='Resume a the last config',
        )

        parser.add_argument(
            '--check_model_size',
            dest='check_model_size',
            action='store_true',
            help='Run a check for the size of the model and data',
        )

        parser.add_argument(
            '-at',
            '--start_at_scene',
            dest='start_scene',
            default=None,
            help='The name of the scenario that we should start from.',
        )

        parser.add_argument(
            '--test_at_start',
            dest='test_at_start',
            action='store_true',
            help='If we should start with running a test with the scene name @pre_run.' +
            ' This is useful together with the resume and new data in the test/validate that we' +
            ' want to run through the software.',
        )

        parser.add_argument(
            '--skip_tb_copy',
            action='store_true',
            dest='skip_tensorboard',
            help='Skip tensorboard copy if you are resuming previious run',
        )

        parser.add_argument(
            '--save_train_visuals',
            action='store_true',
            dest='save_train_visuals',
            help='Save training visuals to tensorboard. This increases the size of the the tensorboard' +
            ' significantly and should be used sparingly',
        )

        parser.add_argument(
            '-batch_size',
            action='store',
            dest='batch_size',
            type=int,
            help='Batch size. Note that if you want to shrink the size during different scenes you can use the config' +
            'batch_size_multipler for task or scene config',
        )

        # More than 2 workers seem to cause shared memory error in Docker and the speed is hardly much better
        parser.add_argument(
            '-num_workers',
            action='store',
            dest='num_workers',
            default=4,
            type=int,
            help='Number of workers to use in PyTorch DataLoader',
        )

        parser.add_argument(
            '-test_run',
            dest='test_run',
            action='store_true',
            help='Runs a test run through the code with scene using minimal epoch_size, epochs & repeat',
        )

        parser.add_argument(
            '-silent_init',
            dest='silent_init_info',
            action='store_true',
            help='Skip all the initialization info',
        )

        parser.add_argument(
            '-graph_info',
            dest='graph_info',
            action='store_true',
            help='Skip all the graph info at the start of each run',
        )

        parser.add_argument(
            '--activate_torch_cuda_benchmark',
            dest='activate_torch_cuda_benchmark',
            action='store_true',
            help='Teoretically the "torch.backends.cudnn.benchmark = True" improves performance.' +
            ' Unfortunately this is not always the case when input size varies and the memory footprint increases.')

        parser.add_argument('--pseudo_mask_margin',
                            dest='pseudo_mask_margin',
                            default=0.5,
                            type=positive_float,
                            help='''
                            The margin used in margin loss for setting the boundary for when to accept predictions for
                            the pseudo labels (i.e. teacher\'s labels).
                            ''')

        parser.add_argument('--pseudo_loss_factor',
                            dest='pseudo_loss_factor',
                            default=0.5,
                            type=positive_float,
                            help='''
                            The factor to multiply the pseudo loss with. If we use a higher margin it can possibly be
                            of interest to increase this factor as the labels will have less noise in them.
                            ''')

        parser.add_argument(
            '--tag',
            dest='tag',
            default=None,
            help='A tag for the current run, e.g. "0.7_signal_to_noise_for_complications"',
        )

        parser.add_argument('--version', action='version', version='%(prog)s 0.6')

        return parser.parse_args()

    def get(self, key, default=None):
        if (key in self.cfgs):
            return self.cfgs[key]

        return default

    def read_environment_variables(self):
        required_environment_variables = ['TENSOR_BACKEND', 'DEVICE_BACKEND', 'IMAGE_BACKEND']
        for var in required_environment_variables:
            if var not in os.environ:
                raise IndexError(f'You forgot to set {var} in your environment, i.e. {var}="your value"')
            self.cfgs[var] = os.environ[var]

        optional_environment_variables = [
            'ANNOTATION_ROOT',
            'LOG_ROOT',
            'TMP_ROOT',
            'MODEL_ZOO_ROOT',
            'LOG_ROOT',
            'TMP_ROOT',
            'MODEL_ZOO_ROOT',
            'IMAGENET_ROOT',
            'CIFAR10_ROOT',
            'XRAY_ROOT',
            'COCO_ROOT',
        ]
        for var in optional_environment_variables:
            if var in os.environ:
                self.cfgs[var.lower()] = os.environ[var]
            else:
                self.cfgs[var.lower()] = None

    def prep_resume(self):
        ui = Console_UI()
        resume_prefix = self.get('resume')
        resume_scene = self.get('resume_scene')
        if resume_scene is not None and resume_prefix is None:
            raise ValueError('You must provide resume prefix if you have set a resume scene')

        # for debug mode uncomment:
        # scenario_log_root = "/media/max/SSD_1TB/log/"
        if resume_prefix.lower() == 'last':
            dirs = sorted([d for d in iglob(f'{self.scenario_log_root}/*/*/neural_nets')])
            dirs = [d for d in dirs if len([f for f in iglob(f'{d}/*{resume_scene}.t7')]) > 0]
            if len(dirs) == 0:
                raise ValueError(f'No previous runs found in \'{self.scenario_log_root}\' with *{resume_scene}.t7')
            resume_prefix = dirs[-1].lstrip(self.scenario_log_root).rstrip('/neural_nets')

            ui.inform_user(f'Resuming run from {resume_prefix}')
        elif resume_prefix is not None:
            resume_prefix = retrieve_dir(path=resume_prefix, base_path=self.scenario_log_root, expected_depth=1)

            if resume_scene != "last":
                matches = [f for f in iglob(f'{self.scenario_log_root}/{resume_prefix}/neural_nets/*{resume_scene}.t7')]
                if len(matches) == 0:
                    err_msg = f'Could not find scene "{resume_scene}" for "{resume_prefix}"'
                    last_matches = [f for f in iglob(f'{self.scenario_log_root}/{resume_prefix}/neural_nets/*last.t7')]
                    if len(last_matches) > 0:
                        base = basename(last_matches[0])
                        base = base[0:-len("last.t7")]
                        possible_matches = ', '.join([
                            basename(f)[(len(base)):-len(".t7")]
                            for f in iglob(f'{self.scenario_log_root}/{resume_prefix}/neural_nets/{base}*.t7')
                        ])

                        err_msg = f'{err_msg} - did you mean any of the following scenes: {possible_matches}'
                    raise ValueError(err_msg)

                ui.inform_user(f'Resuming run from {resume_prefix} @ {resume_scene}')
            else:
                ui.inform_user(f'Resuming run from {resume_prefix}')

        if resume_scene != "last" and resume_prefix is None:
            raise ValueError(f'Can not have a resume scene {resume_scene} but no resume directory')

        self.cfgs['resume'] = resume_prefix
        # for debug mode uncomment:
        # self.cfgs['resume'] = "../%s" % self.cfgs['resume']
        if not self.cfgs['skip_tensorboard']:
            dst_tensorboard_path = os.path.join(self.log_folder, 'tensorboard')
            if os.path.exists(dst_tensorboard_path):
                ui.inform_user(f'Removing previous tensorboard catalogue: {dst_tensorboard_path}')
                shutil.rmtree(dst_tensorboard_path)

            ui.inform_user('Copying the previous tensorboard data')
            shutil.copytree(
                src=os.path.join(self.scenario_log_root, resume_prefix, 'tensorboard'),
                dst=dst_tensorboard_path,
            )
