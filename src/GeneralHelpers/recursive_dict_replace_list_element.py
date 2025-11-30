from typing import Dict, Union, List


def recursive_dict_replace_list_element(source: Dict[str, Union[str, dict, int, float]],
                                        element_id: str,
                                        replace_with: List[str],
                                        debug_path: str = '/'):
    if not isinstance(source, dict):
        raise ValueError(f'The element at {debug_path} is not a proper dictionary')

    ret = {}
    for key in source.keys():
        if key == element_id:
            raise IndexError(f'Expected key {element_id} to be a list element, not a key in dictionary ({debug_path})')

        element = source[key]
        if isinstance(element, dict):
            try:
                ret[key] = recursive_dict_replace_list_element(source=element,
                                                               element_id=element_id,
                                                               replace_with=replace_with,
                                                               debug_path=f'{debug_path}{key}/')
            except ValueError as e:
                print(element)
                raise e

        elif type(element) == tuple or type(element) == list:
            if len([v for v in element if v == element_id]) == 0:
                ret[key] = element
            else:
                clean_vars = [v for v in element if v != element_id]
                ret[key] = [*clean_vars, *replace_with]
        else:
            ret[key] = element
    return ret
