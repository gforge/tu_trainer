import pandas as pd


def flatten_dict(dd, separator='_', prefix=''):
    return {
        prefix + separator + k if prefix else k: v for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {
        prefix: dd
    }


def merge_memory(d1, d2):
    if d2 is None:
        return d1
    for key in ['param', 'total']:
        if d1[key] is None:
            d1[key] = d2[key]
        elif d2[key] is not None:
            d1[key] += d2[key]
    return d1


class MemoryProfiler():

    def __get_model_memory_usage(self, model_name: str):
        encoder_memory = {'param': 0, 'total': 0}
        model = self.graphs[self.train_set_name].models[model_name]
        if model.encoder is not None:
            if hasattr(model.encoder, 'network_memory_usage'):
                merge_memory(encoder_memory, model.encoder.network_memory_usage)
            elif hasattr(model.encoder, 'neural_nets'):
                for n in model.encoder.neural_nets:
                    merge_memory(encoder_memory, n.network_memory_usage)

        decoder_memory = {'param': 0, 'total': 0}
        if model.decoder is not None:
            if hasattr(model.decoder, 'network_memory_usage'):
                merge_memory(decoder_memory, model.decoder.network_memory_usage)
            elif hasattr(model.decoder, 'neural_nets'):
                for n in model.decoder.neural_nets:
                    merge_memory(decoder_memory, n.network_memory_usage)

        return {'encoder': encoder_memory, 'decoder': decoder_memory}

    def get_memory_usage_profile(self):
        usage = {}

        for model_name in self.graphs[self.train_set_name].models.keys():
            usage[model_name] = self.__get_model_memory_usage(model_name=model_name)

        usage_dataset = pd.DataFrame({k: flatten_dict(usage[k]) for k in usage.keys()}).transpose()
        usage_dataset['total'] = usage_dataset['encoder_total'] + usage_dataset['decoder_total']
        usage_dataset['param_total'] = usage_dataset['encoder_param'] + usage_dataset['decoder_param']
        usage_dataset = usage_dataset.sort_values(by="total", ascending=False)

        return usage_dataset
