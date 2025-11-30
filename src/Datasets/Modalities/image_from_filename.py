import os
import numpy as np
from DataTypes import SpatialTransform
from Utils.ImageTools import get_random_transformed_image, get_fixed_transformed_image, \
    load_image, fix_dimension_and_normalize
from global_cfgs import Global_Cfgs

from .Base_Modalities.base_input import Base_Input
from .Base_Modalities.base_image import Base_Image
from .Base_Modalities.base_csv import Base_CSV


class Image_from_Filename(Base_Image, Base_Input, Base_CSV):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._spatial_transform == SpatialTransform.random:
            self.get_transformed_image = get_random_transformed_image
        elif self._spatial_transform == SpatialTransform.fix:
            self.get_transformed_image = get_fixed_transformed_image
        else:
            raise KeyError(f'Unknown spatial transform: {self._spatial_transform}')

        self.im_root = self.__get_img_root()
        assert self.im_root is not None, f'Image root hasn\'t been set for {self.dataset_name.lower()}'
        if not os.path.exists(self.im_root):
            raise FileNotFoundError(f'The dataset {self.dataset_name.lower()} can\'t be found in configs')

        self.__hit = 0
        self.__miss = 0

        self.output_size = (self.height, self.width)

    def __get_img_root(self):
        img_root = self._cfgs.img_root
        if img_root is None:
            img_root = f'{self.dataset_name.lower()}_root'
        return Global_Cfgs().get(img_root)

    def has_discriminator_loss(self):
        return True

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'Implicit',
            'consistency': self.get_consistency(),
        }

    def get_item(self, index, num_views=None):
        filenames = self.get_content(index)

        spatial_transforms = None

        ims = np.zeros(
            (len(filenames), self.num_jitters, self.num_channels, self.height, self.width),
            dtype=np.float32,
        )

        for sub_index in range(len(filenames)):
            filename = filenames[sub_index]
            im, success = self._load_image(filename)

            for j in range(self.num_jitters):
                tr_im, spatial_transform = self.get_transformed_image(im=im,
                                                                      output_size=self.output_size,
                                                                      augmentation_index=j,
                                                                      skip_pad=self._cfgs.skip_padding,
                                                                      skip_resize=False)
                ims[sub_index, j, :, :, :] = tr_im

                spnumpy = spatial_transform.to_numpy()
                if spatial_transforms is None:
                    spatial_transforms = np.zeros(
                        (len(filenames), self.num_jitters, *spnumpy.shape),
                        dtype=spnumpy.dtype,
                    )
                spatial_transforms[sub_index, j] = spnumpy
            if success:
                self.__hit += 1
            else:
                self.__miss += 1

        return {self.get_batch_name(): ims, 'spatial_transforms': spatial_transforms}

    def _load_image(self, filename):
        im_path = os.path.join(self.im_root, filename.lstrip('/'))
        im, success = load_image(im_path)
        im, _ = fix_dimension_and_normalize(im=im,
                                            keep_aspect=self.keep_aspect,
                                            scale_to=self.scale_to,
                                            colorspace=self.colorspace)
        return im, success

    def get_reconstruction_loss_name(self):
        return '%s_l1_reconst' % self.get_name()

    def get_reconstruction_loss_cfgs(self):
        return {
            'loss_type': 'l1_laplacian_pyramid_loss',
            'modality_name': self.get_name(),
            'output_name': self.get_decoder_name(),
            'target_name': self.get_batch_name(),
            'num_channels': self.get_channels(),
            'pyramid_levels': 3,
            'sigma': 1.,
            'kernel_size': 5
        }

    def get_discriminator_loss_name(self):
        return f'{self.get_name()}_real_fake_disc'

    def get_discriminator_loss_cfgs(self):
        return {
            'loss_type': 'wGAN_gp',
            'modality_name': self.get_name(),
            'real_name': self.get_batch_name(),
            'fake_name': self.get_decoder_name(),
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'Cascade',
                'block_type': 'Basic',
                'add_max_pool_after_each_block': True,
                'blocks': [{
                    'output_channels': 32,
                    'no_blocks': 1,
                    'kernel_size': 5,
                }, {
                    'output_channels': 64,
                    'no_blocks': 1,
                    'kernel_size': 3,
                }],
                'consistency': self.get_consistency(),
            }
        }

    def set_runtime_value(self, runtime_value_name, value, indices, sub_indices=None, subgroup_name=None):
        pass  # Not really required to save the image
