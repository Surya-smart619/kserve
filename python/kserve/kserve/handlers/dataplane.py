from kserve.model_repository import ModelRepository
from kserve.model import ModelType


class DataPlane:

    def __init__(self, model_registry: ModelRepository):
        self._model_registry = model_registry

    async def model_metadata(self, name):
        # TODO: Make await optional for sync methods
        model = self._model_registry.get_model(name)
        return await model.metadata()

    async def explain(self, payload, name):
        model = self._model_registry.get_model(name)
        response = await model(payload, model_type=ModelType.EXPLAINER)
        return response

    async def infer(
        self, payload, name: str
    ):
        model = self._model_registry.get_model(name)
        prediction = await model.predict(payload)
        return prediction
