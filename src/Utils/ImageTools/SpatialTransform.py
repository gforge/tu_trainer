import numpy as np

from GeneralHelpers import init_nan_array, extract_and_set_shape, append_shape, unwrap
from .pad_and_center_image import PaddingDetails, IndexeableShape


class SpatialTransform():

    def __init__(
        self,
        M: np.ndarray,
        cropped_im_shape: IndexeableShape,
        pre_transform_shape: IndexeableShape,
        padding: PaddingDetails,
    ):
        super().__init__()
        self.__M = M
        self.__padding = padding
        self.__pre_transform_shape = pre_transform_shape
        self.__cropped_im_shape = cropped_im_shape

    def to_numpy(self) -> np.ndarray:
        ret = init_nan_array(shape=(3 + 1 + 1 + 4, 4), dtype=np.float64)
        ret[0:3, 0:3] = self.__M

        append_shape(destination=ret, shape=self.__pre_transform_shape, main_idx=3)
        append_shape(destination=ret, shape=self.__cropped_im_shape, main_idx=4)
        ret[5:] = self.__padding.to_numpy()
        return ret

    @staticmethod
    def from_pickled(array: np.ndarray):
        array = unwrap(array)
        args = {'M': array[0:3, 0:3], 'padding': PaddingDetails.from_pickled(array=array[5:])}

        extract_and_set_shape(destination=args, source_array=array, name='pre_transform_shape', main_idx=3)
        extract_and_set_shape(destination=args, source_array=array, name='cropped_im_shape', main_idx=4)

        return SpatialTransform(**args)

    def convert_original_points_to_processed_image_space(self, original_points: np.ndarray) -> np.ndarray:
        """
        Converts the points for the processed image that the network sees into points
        that correspond to the original image.

        @param original_points a numpy array of nx2 coordinates in relative values
        """

        assert original_points.max() <= 1, \
            f'point values must be between 0 to 1 but max value is {original_points.max()}'
        assert original_points.min() >= 0, \
            f'point values must be between 0 to 1 but min value is {original_points.min()}'

        self.__padding.transform_coordinates_2_padded_space(original_points)

        extra_ones = np.ones((original_points.shape[0], 1))
        original_points = np.concatenate((original_points, extra_ones), axis=1)

        original_points[:, 0] *= self.__pre_transform_shape[1]
        original_points[:, 1] *= self.__pre_transform_shape[0]

        tr_points = np.dot(self.__M, original_points.T).T
        tr_points /= np.expand_dims(tr_points[:, 2], 1)
        tr_points = tr_points[:, 0:2]

        tr_points[:, 0] /= self.__cropped_im_shape[1]
        tr_points[:, 1] /= self.__cropped_im_shape[0]

        return tr_points

    def convert_processed_image_points_to_original_space(self, tr_points: np.ndarray) -> np.ndarray:
        """
        Converts the points for the processed image that the network sees into points
        that correspond to the original image.

        @param tr_points a numpy array of nx2 coordinates in relative values
        """

        extra_ones = np.ones((tr_points.shape[0], 1))
        tr_points = np.concatenate((tr_points, extra_ones), axis=1)

        tr_points[:, 0] *= self.__cropped_im_shape[1]
        tr_points[:, 1] *= self.__cropped_im_shape[0]

        original_points = np.dot(np.linalg.inv(self.__M), tr_points.T).T
        original_points /= np.expand_dims(original_points[:, 2], 1)
        original_points = original_points[:, 0:2]

        original_points[:, 0] /= self.__pre_transform_shape[1]
        original_points[:, 1] /= self.__pre_transform_shape[0]

        self.__padding.transform_coordinates_from_padded_space(original_points)

        return original_points
