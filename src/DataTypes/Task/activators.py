from DataTypes.general import BaseModelWithGet


class Task_Cfg_Activators(BaseModelWithGet):
    classification: bool = False
    "Whether to use classification outcomes"

    identification: bool = False
    "Wheter to see if network can identify if two views belong to the same subject"

    reconstruction: bool = False
    "Use a decoder for regularizing the network"

    regression: bool = False
    "Use regression outcomes, i.e. measurements"

    pi_model: bool = False
    "Compare different jitters predictions and make sure they agree with each other"

    real_fake: bool = False
    "For wGAN but not used"
