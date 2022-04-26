import inspect

import tornado.web
from kserve.model import ModelType
from kserve.model_repository import ModelRepository
from ray.serve.api import RayServeHandle


class DataPlane:

    def __init__(self, model_registry: ModelRepository):
        self._model_registry = model_registry

    async def model_metadata(self, name):
        model = self._model_registry.get_model(name)
        return await model.metadata()

    async def explain(self, payload, name):
        model = self._model_registry.get_model(name)
        response = await model(payload, model_type=ModelType.EXPLAINER)
        if not isinstance(model, RayServeHandle):
            response = await model(payload, model_type=ModelType.EXPLAINER)
        else:
            model_handle = model
            response = await model_handle.remote(payload, model_type=ModelType.EXPLAINER)
        return response

    async def infer(
        self, payload, name: str
    ):
        model = self._model_registry.get_model(name)
        prediction = (await model.predict(payload)) if inspect.iscoroutinefunction(model.predict) \
            else model.predict(payload)
        return prediction

    async def list(self):
        return {"models": list(self._model_registry.get_models().keys())}

    async def status(self, name):
        model = self._model_registry.get_model(name)
        if model is None:
            raise tornado.web.HTTPError(
                status_code=404,
                reason="Model with name %s does not exist." % name
            )
        is_ready = self._model_registry.is_model_ready(name)
        return {
            "name": name,
            "ready": is_ready
        }

    async def live(self):
        response = {"status": "alive"}
        return response

    async def load(self, name):
        self._model_registry.load(name)
        return {
            "name": name,
            "load": True
        }

    async def unload(self, name):
        self._model_registry.unload(name)
        return {
            "name": name,
            "unload": True
        }
