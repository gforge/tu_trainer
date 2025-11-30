import numpy as np
import pandas as pd
import time
from typing import Dict, Optional, Any

from .helpers import generate_sampling_bins
from GeneralHelpers import init_nan_array
from torch.utils.data import Dataset
from Datasets.Modalities.Base_Modalities.base_explicit import Base_Explicit


class Batch_Loader(Dataset):
    def __init__(
        self,
        dataset_name: str,
        experiment_name: str,
        batch_size: int,
        view_dropout: Optional[float],
        annotations: pd.DataFrame,
        explicit_modalities: Dict[str, Base_Explicit],
    ):
        assert batch_size > 0
        self.__annotations = annotations
        self.__dataset_name = dataset_name
        self.__experiment_name = experiment_name

        assert view_dropout is None or (
            view_dropout >= 0 and view_dropout < 1
        ), "The view dropout must be None or a probability, i.e. between 0 and 1"
        self.__view_dropout = view_dropout

        self.__bins, self.__bin_weights = generate_sampling_bins(
            annotations=self.__annotations,
            batch_size=batch_size,
        )
        assert (
            len(self) > 0
        ), f"Failed to generate bins for {self} with batch size {batch_size}"

        self.__explicit_modalities = explicit_modalities

    def __repr__(self):
        return f"Loader {self.__dataset_name} >> {self.__experiment_name} ({len(self.__annotations)} @ {len(self)})"

    def __len__(self):
        return len(self.__bins)

    def __getitem__(self, index) -> Dict[str, Any]:
        batch = {
            "post_batch_hooks": set(),
            "time": {
                "start": time.time(),
                "load": {},
                "process": {},
                "encode": {},
                "decode": {},
                "backward": {},
            },
        }
        batch.update(self.__get_indices(index=index))

        # It is very important to first load the images, then load other modalities

        batch_data = [
            modality.get_batch(batch)
            for modality_name, modality in self.__explicit_modalities.items()
        ]

        try:
            for data in batch_data:
                batch.update(data)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve batch for {self}: {e}")

        return batch

    def __get_indices(self, index: int):
        indices = self.__bins[index]

        num_views = np.array([len(self.__annotations.loc[i]) for i in indices], dtype='uint8')
        sub_indices = [list(self.__annotations.loc[i].index.get_level_values(0)) for i in indices]
        if self.__view_dropout is not None:
            for i, si in enumerate(sub_indices):
                select_cases = np.random.rand(len(si)) > self.__view_dropout
                if np.any(select_cases):
                    sub_indices[i] = [sub_indices[i][ii] for ii, v in enumerate(select_cases) if v]

        sub_indices_shape = (len(indices), max([len(s) for s in sub_indices]))
        sub_indices_np = init_nan_array(shape=sub_indices_shape)
        for i, si in enumerate(sub_indices):
            sub_indices_np[i, 0:len(si)] = si

        current_batch_size = self.__bin_weights[index]
        batch_size_info = f"{current_batch_size} ({len(self.__bins[index])} bins)"
        if self.__view_dropout is not None:
            batch_size_info = f"{batch_size_info} before dropping {self.__view_dropout*100:0.0f}% of views"

        return {
            "indices": indices,
            "sub_indices": sub_indices_np,
            "num_views": num_views,
            "current_batch_size": current_batch_size,
            "batch_size_info": batch_size_info,
            "batch_index": index,
            "epoch_size": len(self),
        }
