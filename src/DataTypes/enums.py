from enum import Enum


class LossWeightType(str, Enum):
    basic = 'basic'
    sqrt = 'sqrt'
    max = 'max'


class Consistency(str, Enum):
    number = 'Number'
    d1 = '1D'
    d2 = '2D'
    d3 = '3D'


class Purpose(str, Enum):
    encoder = 'encoder'
    decoder = 'decoder'


class OptimizerType(Enum):
    SGD = 'SGD'
    "Stochastic gradient descent"

    ASGD = 'ASGD'
    "Averaged stochastic gradient descent"

    Adam = 'Adam'
    "Adam optimizer"

    Adagrad = 'Adagrad'
    "Adagrad optimizer"

    Adadelta = 'Adadelta'
    "Adadelta optimizer"

    AdamW = 'AdamW'
    "AdamW optimizer"

    RMSprop = 'RMSprop'
    "RMSprop optimizer"

class JitterPool(Enum):
    max = 'max'
    mean = 'mean'


class TaskType(Enum):
    training = 'training'
    validation = 'validation'
    test = 'test'


class Learning_Rate_Type(Enum):
    decay = 'decay'


class Learning_Rate_Function(Enum):
    cosine = 'cosine'
    "Learning rate changes with the cosine function"

    linear = 'linear'
    "Learning rate is linearly changing throghout the epoch"


class SpatialTransform(Enum):
    random = 'random'
    fix = 'fix'


class Colorspace(Enum):
    rgb = 'RGB'
    gray = 'Gray'
