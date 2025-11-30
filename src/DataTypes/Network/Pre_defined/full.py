from ..base import Network_Cfg_Base

from .raw import Network_Cfg_Pre_Defined_Raw


class Network_Cfg_Pre_Defined(Network_Cfg_Pre_Defined_Raw, Network_Cfg_Base):

    def __str__(self) -> str:
        return f'{self.model} @ {self.repo_or_dir}'

    def get_id(self) -> str:
        """Retrieves an id based on the repo and model

        Returns:
            str: A string without unexpected characters
        """
        repo_clean = self.repo_or_dir.replace("/", "_").replace(":", "_")
        model = self.model.replace("/", "_").replace(":", "_")

        return f'{repo_clean}_{model}'
