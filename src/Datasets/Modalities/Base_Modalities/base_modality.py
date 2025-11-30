import traceback
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Generic
from torch import Tensor

from UIs.console_UI import Console_UI
from DataTypes import SpatialTransform, Consistency, Modality_Type


class Base_Modality(Generic[Modality_Type], metaclass=ABCMeta):

    def is_explicit_modality(self):
        # this function will be overridden in Base_Explicit
        return False

    def is_implicit_modality(self):
        # this function will be overridden in Implicit
        return False

    def is_input_modality(self):
        # this function will be overridden in Base_Input
        return False

    def is_output_modality(self):
        # this function will be overridden in Base_Output
        return False

    def __init__(
        self,
        dataset_name: str,
        min_channels: int,
        modality_name: str,
        modality_cfgs: Modality_Type,
        spatial_transform: SpatialTransform,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.modality_name = modality_name.lower()
        self._cfgs = modality_cfgs
        self._min_channels = min_channels
        self._spatial_transform = spatial_transform
        self.__batch_loss = []

        if self._cfgs.tensor_shape:
            self.set_tensor_shape(self._cfgs.tensor_shape)

    @property
    def consistency(self) -> Consistency:
        return self._cfgs.consistency

    @consistency.setter
    def age(self, value: Consistency):
        if self.consistency == value:
            return

        if self.consistency is not None:
            raise ValueError(f'Conflicting consistencire in {self.name}:' +
                             f' - config says {self.consistency} while trying to change to {value}')

        self._cfgs.consistency = value

    @property
    def num_jitters(self):
        return self._cfgs.num_jitters

    def has_reconstruction_loss(self):
        return False

    def has_identification_loss(self):
        return False

    def has_discriminator_loss(self):
        return False

    def has_classification_loss(self):
        return False

    def has_regression_loss(self):
        return False

    def __append_loss(self, batch_loss):
        if isinstance(batch_loss, Tensor):
            batch_loss = batch_loss.detach().item()
        self.__batch_loss.append(batch_loss)

    def analyze_results(self, batch, loss):
        self.__append_loss(loss)
        try:
            self.analyze_modality_specific_results(batch)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            Console_UI().warn_user(f'Failed to get results for {self.modality_name}: {e}')

    def analyze_modality_specific_results(self, batch):
        pass  # Implemented in subclass if any

    def report_epoch_summary(self, summary):
        summary['modalities'][self.get_name()] = {}
        if len(self.__batch_loss) > 0:
            summary['modalities'][self.get_name()]['loss'] = np.mean(self.__batch_loss)
            self.__batch_loss = []

        self.report_modality_specific_epoch_summary(summary)

    def report_modality_specific_epoch_summary(self, summary):
        pass  # Implemented in subclass if any

    def get_name(self):
        return self.modality_name.lower()

    def get_encoder_name(self):
        return f'encoder_{self.get_name()}'

    def get_decoder_name(self):
        return f'decoder_{self.get_name()}'

    def get_real_name(self):
        return self.get_batch_name()

    def get_fake_name(self):
        return self.get_decoder_name()

    def get_batch_name(self):
        return self.get_encoder_name()

    def get_target_name(self):
        return self.get_batch_name()

    def get_modality_cfgs(self):
        base_dict = self._cfgs.dict()
        base_dict['tensor_shape'] = self.get_tensor_shape()
        base_dict['encoder_name'] = self.get_encoder_name()
        base_dict['decoder_name'] = self.get_decoder_name()
        base_dict['batch_name'] = self.get_batch_name()
        base_dict['target_name'] = self.get_target_name()
        base_dict['has_reconstruction_loss'] = self.has_reconstruction_loss()
        base_dict['has_classification_loss'] = self.has_classification_loss()
        base_dict['has_identification_loss'] = self.has_identification_loss()
        base_dict['has_discriminator_loss'] = self.has_discriminator_loss()

        return base_dict

    def get_shape(self):
        return [self.num_jitters, *self.get_tensor_shape()]

    def set_tensor_shape(self, shape):
        if len(shape) == 1:
            self.set_channels(shape[0])
        if len(shape) == 2:
            self.set_channels(shape[0])
            self.set_width(shape[1])
        if len(shape) == 3:
            self.set_channels(shape[0])
            self.set_width(shape[1])
            self.set_height(shape[2])
        if len(shape) == 4:
            self.set_channels(shape[0])
            self.set_width(shape[1])
            self.set_height(shape[2])
            self.set_depth(shape[3])

    def get_tensor_volume(self):
        return np.prod(self.get_tensor_shape())

    def get_shape_str(self):
        return 'x'.join(str(d) for d in self.get_shape())

    def set_consistency(self, consistency: Consistency):
        self.consistency = consistency

    def get_consistency(self) -> Consistency:
        return self.consistency

    @abstractmethod
    def get_tensor_shape(self):
        """
        this function will be overridden in
        Base_Number, Base_Sequence, Base_Plane and Base_Volume"
        """
        pass

    def is_classification(self):
        return False

    def is_regression(self):
        return False
