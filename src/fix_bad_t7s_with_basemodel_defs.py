#!/usr/bin/env python

import argparse
import os
import torch
from pydantic import BaseModel

from Utils.safe_load_broken_t7_files import safe_load_broken_t7_files

parser = argparse.ArgumentParser(description="""A script for fixing save with Pydantic classes

                   In FileManager the:
                   torch.save(
                       {
                           'state_dict': neural_net.layers.state_dict(),
                           'neural_net_cfgs': neural_net.get_network_cfgs().dict()
                       }, neural_net_path)

                   Didn't contain the `dict()` to the configs and hence loading required access to
                   the DataTypes folder. This script re-saves the same models without the Pydantic
                   dataclasses.
                   """)

parser.add_argument('path', type=str, nargs='+', help='The path or files that we want to convert')

args = parser.parse_args()


def fix_file(file):
    print(f'Fixing {file}')
    try:
        model = torch.load(file)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f'Missing class when loading file {file}: {e}')

    if isinstance(model['neural_net_cfgs'], BaseModel):
        model['neural_net_cfgs'] = model['neural_net_cfgs'].dict()
        torch.save(model, file)
    else:
        print(f'Net configs OK: {type(model["neural_net_cfgs"])}')


def fix_all_files(path: str):
    if os.path.isfile(path):
        return fix_file(path)

    for root, dirs, files in os.walk(path):
        [fix_file(os.path.join(root, f)) for f in files if f.endswith('.t7')]
        [fix_all_files(d) for d in dirs]


for path in args.path:
    fix_all_files(path)
