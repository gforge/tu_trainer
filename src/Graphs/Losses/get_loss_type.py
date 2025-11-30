from typing import Union, Optional, Dict
# Classification losses
from .cross_entropy_loss import Cross_Entropy_Loss
from .bipolar_margin_loss import Bipolar_Margin_Loss
from .hierarchical_BCE_loss import Hierarchical_BCE_Loss
from .triplet_metric_loss import Triplet_Metric_Loss
# Regression losses
from .mse_loss import MSE_Loss
from .mse_wsp_loss import MSE_WSP_Loss
from .mse_line_loss import MSE_Line_Loss
from .mse_independent_line_loss import MSE_Independent_Line_Loss
# Reconstruction losses
from .L1_laplacian_pyramid_loss import L1_Laplacian_Pyramid_Loss
from .L2_loss import L2_Loss
# Real vs fake losses
from .wgan_gp_loss import Wasserstein_GAN_GP_Loss

RegressionLosses = Union[MSE_Loss, MSE_WSP_Loss, MSE_Line_Loss]
AnyClass = Union[Cross_Entropy_Loss, Bipolar_Margin_Loss, Hierarchical_BCE_Loss, L1_Laplacian_Pyramid_Loss, L2_Loss,
                 Wasserstein_GAN_GP_Loss, Triplet_Metric_Loss, RegressionLosses]

_regression_losses = {
    'mse_loss': MSE_Loss,
    'mse_with_spatial_transform': MSE_WSP_Loss,
    'mse_with_spatial_transform_and_line': MSE_Line_Loss,
    'mse_with_spatial_transform_and_independent_line': MSE_Independent_Line_Loss,
}

_classification_losses = {
    'cross_entropy': Cross_Entropy_Loss,
    'bipolar_margin_loss': Bipolar_Margin_Loss,
    'hierarchical_bce': Hierarchical_BCE_Loss,
    'triplet_metric': Triplet_Metric_Loss,
}

_reconstruction_losses = {
    'l1_laplacian_pyramid_loss': L1_Laplacian_Pyramid_Loss,
    'l2_loss': L2_Loss,
}


def _find_type(needle: str, types: Dict[str, AnyClass], valid_graph: bool, err_explanation: str) -> Optional[AnyClass]:
    for (id, loss_class) in types.items():
        if needle == id.lower():
            if not valid_graph:
                raise ValueError(f'The {loss_class.__name__} only works when {err_explanation}')
            return loss_class

    return None


def get_loss_type(
    loss_type: str,
    graph,
) -> AnyClass:
    """
    Convert loss type to a Loss class. Throws error if loss_type not found or any of the booleans are incompatible
    with the loss type.
    """
    loss_type_lowered = loss_type.lower()
    loss = _find_type(needle=loss_type_lowered,
                      types=_classification_losses,
                      valid_graph=(graph.classification or graph.pi_model),
                      err_explanation='only works when classification or PI-loss is active')
    if loss is not None:
        return loss

    loss = _find_type(needle=loss_type_lowered,
                      types=_reconstruction_losses,
                      valid_graph=graph.reconstruction,
                      err_explanation='only works when reconstrionction is active')
    if loss is not None:
        return loss

    loss = _find_type(needle=loss_type_lowered,
                      types=_regression_losses,
                      valid_graph=graph.regression,
                      err_explanation='only works when regression is active')
    if loss is not None:
        return loss

    if loss_type_lowered == 'wGAN_gp'.lower():
        if not graph.real_fake:
            raise ValueError('The Wasserstein_GAN_GP_Loss only works when graph.real_fake is active')
        return Wasserstein_GAN_GP_Loss

    raise KeyError(f'Unknown loss type: {loss_type}')
