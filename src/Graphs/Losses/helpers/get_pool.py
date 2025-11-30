import re

from .view_pool_max import View_Pool_Max
from .view_pool_mean import View_Pool_Mean
from .view_pool_avg_top_prop import View_Pool_AvgTopProp


def get_pool_network(view_pool: str, dim: int, modality_name: str, keepdim: bool = False):

    if view_pool.lower() == 'mean':
        return View_Pool_Mean(dim=dim, keepdim=keepdim)

    if view_pool.lower() == 'max':
        return View_Pool_Max(dim=dim, keepdim=keepdim)

    if view_pool.lower().startswith("AvgTopPerc".lower()):
        perc_match = re.compile(".*[^0-9]([0-9]+)$")
        if re.match(perc_match, view_pool) is None:
            prop = 0.5
        else:
            prop = float(re.sub(".*[^0-9.]([0-9]+)$", "\\1", view_pool)) / 100

        if prop >= 1 or prop <= 0:
            raise ValueError('Expected the viewpool to end wiht a percentage, got:' +
                             f'{view_pool} that converts to {prop}')

        return View_Pool_AvgTopProp(prop=prop, dim=dim, keepdim=keepdim)

    raise NameError(f'There is no implementation of the view pool {view_pool} (see {modality_name})')
