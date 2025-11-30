import os


class ResultId():

    def __init__(self, name, id, path):
        self.name = name
        self.id = id
        self.path = path

    def __repr__(self) -> str:
        return self.id


class SceneUIManager():

    def get_scene(self):
        # Import here as there otherwise is a cyclic dependency
        from scenario import Scenario
        return Scenario().get_current_scene()

    @property
    def name(self):
        scene = self.get_scene()
        return scene.get_name() if scene is not None else '?'

    def get_result_id(self, ds: str, exp: str, task: str, graph: str):
        scene = self.get_scene()
        if scene is None:
            raise RuntimeError(f'Attempted retrieving {ds}_{exp}_{task}_{graph} before initiating scene!')

        scene_name = scene.get_name()
        return ResultId(name=f'{exp}_{task}_{graph}_epoch',
                        id=f'{scene_name}_{ds}_{exp}_{task}_{graph}_epoch',
                        path=os.path.join(ds, scene_name))
