import numpy as np

from .Base_Modalities.base_input import Base_Input
from .Base_Modalities.base_language import Base_Language
from .Base_Modalities.base_csv import Base_CSV

import Utils.language_tools as language_tools


class Char_Sequence(Base_Input, Base_Language, Base_CSV):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_content()

    def preprocess_content(self):
        self.content = self.content.str.normalize('NFKC')
        if not self.case_sensitive:
            self.content = self.content.str.lower()
        if self.discard_numbers:
            self.content = self.content.str.replace(pat='[0-9]+', repl='$', regex=True)
        self.content = self.content.str.replace(pat=f'[^{self.dictionary}]', repl=' ', regex=True)
        self.content = self.content.str.replace(pat='[ ]+', repl=' ', regex=True).fillna(' ')

    def get_item(self, index, num_views=None):
        sentences = self.get_content(index)
        sentence_shape = [len(sentences), self.num_jitters, self.num_channels, self.sequence_length]
        one_hot_sentences = np.zeros(sentence_shape, dtype='float32')
        for sub_index in range(len(sentences)):
            sentence = sentences[sub_index]
            for j in range(self.num_jitters):
                sentence = language_tools.random_jitter(sentence=sentence,
                                                        alphabet=self.dictionary,
                                                        sentence_length=self.sequence_length)
                one_hot_sentences[sub_index, j, :, :] = self.to_one_hot(sentence)

        return {self.get_batch_name(): one_hot_sentences}

    def get_default_model_cfgs(self):
        return {
            "model_type": "One_to_One",
            "neural_net_cfgs": {
                "neural_net_type": "Cascade",
                "block_type": "Basic",
                "add_max_pool_after_each_block": True,
                "blocks": [{
                    "output_channels": 256,
                    "no_blocks": 1,
                    "kernel_size": 5
                }, {
                    "output_channels": 256,
                    "no_blocks": 1,
                    "kernel_size": 5
                }],
                "consistency": self.get_consistency(),
            }
        }

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'Implicit',
            'consistency': self.get_consistency(),
        }

    def to_one_hot(self, sentence):
        idx = np.array([self.char_to_ix[c] for c in sentence])
        one_hot = np.zeros((len(self.dictionary), len(sentence)), dtype='float32')
        one_hot[idx, np.arange(len(sentence))] = 1
        return one_hot

    def set_runtime_value(
        self,
        runtime_value_name,
        value,
        indices,
        sub_indices,
    ):
        pass  # Not implemented for char sequences

    def has_reconstruction_loss(self):
        return True

    def get_reconstruction_loss_name(self):
        return '%s_l2_reconst' % self.get_name()

    def get_reconstruction_loss_cfgs(self):
        return {
            'loss_type': 'l2_loss',
            'modality_name': self.get_name(),
            'output_name': self.get_decoder_name(),
            'target_name': self.get_batch_name(),
            'relu': True,
            'tensor_shape': self.get_tensor_shape()
        }
