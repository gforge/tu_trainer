from .line_writer import LineWriter


class IndependentLineWriter(LineWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._visual_id = 'Visualization/Independent_lines'
        self.results_output_name = 'independent_lines'
