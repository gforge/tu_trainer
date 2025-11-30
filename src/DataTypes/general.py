from typing import Any
from pydantic import Field, BaseModel
from typing_extensions import Annotated

Proportion = Annotated[float, Field(ge=0, le=1)]


class BaseModelWithGet(BaseModel):

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve an element using string

        Copies the behavior of the dict.get() method

        Args:
            name (str): The string
            default (Any, optional): The default value if none is found. Defaults to None.

        Returns:
            Any: The value if it exists otherwise it returns the default value
        """
        try:
            if not hasattr(self, name):
                return default

            return getattr(self, name)
        except AttributeError:
            return default

    def _extend_with_another_models_data(self, another_model: BaseModel):
        """We want to propagate values into the next config but only if those values
        have been defined.

        Args:
            another_model (BaseModel): Another configuration that we want to pick some values from
        """

        # Use __dict__ to update internal value
        extras = {
            k: another_model.get(k)
            for k in another_model.dict().keys()
            if k in self.__dict__ and self.__dict__[k] is None
        }
        self.__dict__.update(**extras)
