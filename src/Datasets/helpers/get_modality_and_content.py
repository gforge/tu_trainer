from typing import Tuple, TypeVar, Union
import pandas as pd
from DataTypes import Any_Modality_Cfg, Modality_Csv_Column_Prefixes_Cfg, Modality_Csv_Mutliple_Columns_Cfg, \
    Modality_Csv_Single_Column_Cfg, Modality_Image_Cfg, Modality_Implicit_Cfg, \
    Modality_Style_Cfg, Modality_Text_Cfg
from DataTypes import Consistency

from Datasets.Modalities.implicit_number import Implicit_Number
from Datasets.Modalities.implicit_sequence import Implicit_Sequence
from Datasets.Modalities.implicit_plane import Implicit_Plane
from Datasets.Modalities.implicit_volume import Implicit_Volume
from Datasets.Modalities.style_number import Style_Number
from Datasets.Modalities.style_sequence import Style_Sequence
from Datasets.Modalities.style_plane import Style_Plane
from Datasets.Modalities.style_volume import Style_Volume
from Datasets.Modalities.id_from_indices import ID_from_Indices
from Datasets.Modalities.pseudo_label import Pseudo_Label
from Datasets.Modalities.one_vs_rest import One_vs_Rest
from Datasets.Modalities.bipolar import Bipolar
from Datasets.Modalities.multi_bipolar import Multi_Bipolar
from Datasets.Modalities.char_sequence import Char_Sequence
from Datasets.Modalities.image_from_filename import Image_from_Filename
from Datasets.Modalities.hierarchical_label import Hierarchical_Label
from Datasets.Modalities.multi_coordinate import Multi_Coordinate
from Datasets.Modalities.multi_line import Multi_Line
from Datasets.Modalities.multi_independent_line import Multi_Independent_Line
from Datasets.Modalities.multi_regression import Multi_Regression
from Datasets.helpers import Dictionary_Generator

dg = Dictionary_Generator()

Any_Modality = TypeVar('Any_Modality', Implicit_Number, Implicit_Sequence, Implicit_Plane, Implicit_Volume,
                       Style_Number, Style_Sequence, Style_Plane, Style_Volume, ID_from_Indices, Pseudo_Label,
                       One_vs_Rest, Bipolar, Multi_Bipolar, Char_Sequence, Image_from_Filename, Hierarchical_Label,
                       Multi_Coordinate, Multi_Line, Multi_Independent_Line, Multi_Regression)


def _get_multi_bipolar(
    annotations,
    name: str,
    cfgs: Modality_Csv_Mutliple_Columns_Cfg,
):
    columns = Multi_Bipolar.get_csv_column_names(column_defintions=cfgs.columns, modality_name=name)

    return Multi_Bipolar, annotations[columns], dg.get_bipolar_dictionary(modality_name=name,
                                                                          label_dictionary=cfgs.dictionary)


def _get_multi_measurement(
    class_definition,
    annotations,
    name: str,
    cfgs: Modality_Csv_Column_Prefixes_Cfg,
):
    columns = []
    for prefix in cfgs.column_prefixes:
        columns.extend(class_definition.prefix_to_column_names(prefix=prefix))

    for colname in columns:
        assert colname in annotations.columns, f'The {colname} doesn\'t exist among columns in annotation for {name}'

    return class_definition, annotations[columns], None


def _get_multi_regression(
    annotations,
    name: str,
    cfgs: Modality_Csv_Mutliple_Columns_Cfg,
):
    for colname in cfgs.columns:
        msg = f'The {colname} doesn\'t exist among columns in annotation for {name}'
        assert colname in annotations.columns, msg

    return Multi_Regression, annotations[cfgs.columns], None


def _get_implicit(
    annotations,
    name: str,
    cfgs: Modality_Implicit_Cfg,
) -> Tuple[Union[Implicit_Number, Implicit_Sequence, Implicit_Plane, Implicit_Volume], None]:

    dictionary = dg.get(name)
    content = None

    if cfgs.consistency == Consistency.number:
        return Implicit_Number, content, dictionary

    if cfgs.consistency == Consistency.d1:
        return Implicit_Sequence, content, dictionary

    if cfgs.consistency == Consistency.d2:
        return Implicit_Plane, content, dictionary

    if cfgs.consistency == Consistency.d3:
        return Implicit_Volume, content, dictionary

    raise KeyError(f'Unknown consistency: "{cfgs.consistency}"')


def _get_style(
    name: str,
    cfgs: Modality_Style_Cfg,
):
    content = None
    dictionary = dg.get(name)

    if cfgs.consistency == Consistency.number:
        return Style_Number, content, dictionary

    if cfgs.consistency == Consistency.d1:
        return Style_Sequence, content, dictionary

    if cfgs.consistency == Consistency.d2:
        return Style_Plane, content, dictionary

    if cfgs.consistency == Consistency.d3:
        return Style_Volume, content, dictionary

    raise KeyError(f'Unknown style: "{cfgs["consistency"]}"')


_measurement_classes = {
    'Multi_Coordinate': Multi_Coordinate,
    'Multi_Line': Multi_Line,
    'Multi_Independent_Line': Multi_Independent_Line
}


def get_modality_and_content(  # NOSONAR - complexity warning
        annotations,
        name: str,
        cfgs: Any_Modality_Cfg,
        ignore_index: int,
) -> Any_Modality:
    if cfgs.type == 'Multi_Bipolar':
        return _get_multi_bipolar(annotations=annotations, name=name, cfgs=cfgs)

    for (id, class_definition) in _measurement_classes.items():
        if cfgs.type == id:
            return _get_multi_measurement(class_definition=class_definition,
                                          annotations=annotations,
                                          name=name,
                                          cfgs=cfgs)

    if cfgs.type == 'Multi_Regression':
        return _get_multi_regression(annotations=annotations, name=name, cfgs=cfgs)

    if cfgs.type == 'Implicit':
        return _get_implicit(annotations=annotations, name=name, cfgs=cfgs)

    if cfgs.type == 'Style':
        return _get_style(name=name, cfgs=cfgs)

    dictionary = dg.get(name)
    if cfgs.type == 'ID_from_Indices':
        return ID_from_Indices, pd.Series(ignore_index, index=annotations.index, dtype='int64'), dictionary

    content = None
    if cfgs.type.lower() == 'Pseudo_Label'.lower():
        return Pseudo_Label, content, dictionary

    if isinstance(cfgs, (Modality_Csv_Single_Column_Cfg, Modality_Text_Cfg, Modality_Image_Cfg)):
        if cfgs.column_name not in annotations:
            available_cols = ", ".join(str(i) for i in list(annotations))
            raise KeyError(f'Unknown column "{cfgs.column_name}" - in the dataset with cols: {available_cols}')

        content = annotations[cfgs.column_name]
        if cfgs.type == 'One_vs_Rest':
            return One_vs_Rest, content, dictionary

        if cfgs.type == 'Bipolar':
            return Bipolar, content, dg.get_bipolar_dictionary(
                modality_name=name,
                label_dictionary=cfgs.dictionary,
            )

        if cfgs.type == 'Char_Sequence':
            return Char_Sequence, content, dictionary

        if cfgs.type == 'Image_from_Filename':
            return Image_from_Filename, content, dictionary

        if cfgs.type == 'Hierarchical_Label':
            return Hierarchical_Label, content, dictionary

    raise KeyError(f'Unknown column type: "{cfgs.type}" for "{name}"')
