import torch
import gc
import time
import re
import os
from typing import List
import objgraph

from .Singleton import Singleton


class _MemoryElement():

    def __init__(self, tensor_object: torch.Tensor, time_desc: str):
        self.id = str(id(tensor_object))
        self.type = type(tensor_object)
        self.size = tuple(tensor_object.size())
        self.value_content = self.size
        self.raw_value = '?'
        if self.is_atomic:
            if len(self.size) == 0:
                self.raw_value = tensor_object.item()
            else:
                self.raw_value = tensor_object[0].item()

            if isinstance(self.raw_value, float):
                self.raw_value = f'{self.raw_value:0.1f}'

            self.value_content = f'{self.raw_value} - value'

        self.is_encoded_image = len(tensor_object.shape) == 5
        self.time_desc = time_desc
        self.time = time.time()
        self.device = tensor_object.device
        self.is_cuda = tensor_object.is_cuda

    def __repr__(self) -> str:
        time_info = f'{time.time() - self.time:.1f} ({self.time_desc})'
        return f'{self.type} - {self.value_content} time info: {time_info} device: {self.device}'

    def get_original(self):
        for obj in gc.get_objects():
            if id(obj) == self.id:
                return obj

        return None

    @property
    def is_created_at_step(self) -> bool:
        return re.match("Step", self.time_desc, flags=re.IGNORECASE) is not None

    @property
    def is_atomic(self) -> bool:
        return len(self.size) == 0 or len(self.size) == 1 and self.size[0] == 1


def print_size(bytes):
    if bytes < 1024:
        return f'{bytes} bytes'

    bytes /= 1024.0
    if bytes < 1024 / 2:
        return f'{bytes} kb'

    bytes /= 1024.0
    if bytes < 1024 / 2:
        return f'{bytes} Mb'

    bytes /= 1024.0
    return f'{bytes} Gb'


def get_memory_snapshot(time_desc: str):
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()

    def is_tensor_object(obj):
        try:
            if torch.is_tensor(obj=obj):
                return True

            if hasattr(obj, 'data') and torch.is_tensor(obj.data):
                return True
        except:  # noqa: E722 #NOSONAR
            return False

        return False

    snapshot = []
    for obj in gc.get_objects():
        try:
            if is_tensor_object(obj):
                snapshot.append(_MemoryElement(obj, time_desc=time_desc))
        except:  # noqa: E722 #NOSONAR
            pass
    return snapshot


class MemorySupervision(metaclass=Singleton):
    """
    A utility class that allows us to monitor where the memory goes
    """

    def __init__(self, multiplicatiohn_threshold=1.05, print_dropped_memory_objects: bool = True) -> None:
        """
        @para multiplicatiohn_threshold tells how much we should allow the memory to expand
                                        before pausing
        """
        super().__init__()
        self.last_memory_usage = None
        self.last_memory_name = None
        self.last_snapshot = get_memory_snapshot(time_desc='Init')
        self.multiplicatiohn_threshold = multiplicatiohn_threshold
        self.print_dropped_memory_objects = print_dropped_memory_objects

    def __has_memory_increased(self, current_usage: int) -> bool:
        if self.last_memory_usage is None:
            return False

        return current_usage > self.last_memory_usage * self.multiplicatiohn_threshold

    def check_usage(self, name: str):
        current_usage = torch.cuda.memory_allocated()
        if self.__has_memory_increased(current_usage=current_usage):
            print(f'Memory has increased from {print_size(self.last_memory_usage)} @ {self.last_memory_name} ->' +
                  f' {print_size(current_usage)} @ {name}')
            res = input("Press enter to continue or 'm + enter' for snapshot analysis...")
            if res == 'm':
                old, new = self.update_memory_snapshot(time_desc=name)
                self.print_usage(elements_to_print=old, title='Old elements')
                self.print_usage(elements_to_print=new, title='New elements')
                input("Press enter to continue...")

        self.last_memory_usage = current_usage
        self.last_memory_name = name
        self.update_memory_snapshot(time_desc=name)

    def update_memory_snapshot(self, time_desc: str = None):
        # We want to keep memory elements as they are marked with time of creation
        new_elements = get_memory_snapshot(time_desc=time_desc)
        if self.print_dropped_memory_objects:
            self.print_usage([*filter(lambda x: not any(x.id == ne.id for ne in new_elements), self.last_snapshot)],
                             title='Dropped memory elements')

        remaining_elements = [*filter(lambda e: any(e.id == ne.id for ne in new_elements), self.last_snapshot)]
        new_elements = [*filter(lambda e: not self.__has_snapshot_element(e), new_elements)]
        self.last_snapshot = [*remaining_elements, *new_elements]
        return (remaining_elements, new_elements)

    def print_interesting_vars_in_step(self, filter_fn=lambda e: e.is_encoded_image):
        images = [*filter(filter_fn, self.last_snapshot)]
        images_in_step = [*filter(lambda e: e.is_created_at_step, images)]
        if len(images_in_step) == 0:
            print('No step images found')
            input("Press enter to continue...")
            return

        for idx, img in enumerate(images_in_step):
            from global_cfgs import Global_Cfgs

            fn = f'{idx}_backref.dot'
            path = os.path.join(Global_Cfgs().log_folder, fn)

            objgraph.show_backrefs(img.get_original(), max_depth=3, filename=path, too_many=50)

    def print_usage(self,
                    elements_to_print: List[_MemoryElement] = None,
                    only_images: bool = False,
                    only_cuda: bool = True,
                    title: str = None) -> None:
        if elements_to_print is None:
            elements_to_print = get_memory_snapshot('New')

        if only_images:
            elements_to_print = [*filter(lambda x: x.is_encoded_image, elements_to_print)]

        if only_cuda:
            elements_to_print = [*filter(lambda x: x.is_cuda, elements_to_print)]

        if title is not None:
            print(title)

        if len(elements_to_print) == 0:
            print('No elements to print')

        atomic_values = set()
        atomic_count = 0
        for e in elements_to_print:
            if e.is_atomic:
                atomic_count += 1
                atomic_values.add(e.raw_value)
            else:
                print(e)
        if atomic_count > 0:
            print(f'Got {atomic_count} atomic values: {atomic_values}')

        images = [*filter(lambda x: x.is_encoded_image, elements_to_print)]
        if len(images) > 0:
            print('Images:')
            for e in images:
                print(e)

        print('\n')

    def __has_snapshot_element(self, element: _MemoryElement) -> bool:
        for existing in self.last_snapshot:
            if existing.id == element.id:
                return True

        return False
