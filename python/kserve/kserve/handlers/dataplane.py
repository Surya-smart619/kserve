import inspect

import tornado.web
from kserve.model import ModelType
from kserve.model_repository import ModelRepository
from ray.serve.api import RayServeHandle


class DataPlane:

    def __init__(self, model_registry: ModelRepository):
        self._model_registry = model_registry
        self._server_name = "kserve"
        self._server_version = "v0.8.0"

    async def metadata(self):
        return {
            "name": self._server_name,
            "version": self._server_version
        }

    async def model_metadata(self, name: str, version: str = None):
        model = self.get_model_from_registry(name, version)
        return await model.metadata()

    async def explain(self, payload, name: str, version: str = None):
        model = self.get_model_from_registry(name, version)
        response = await model.explain(payload)
        if not isinstance(model, RayServeHandle):
            response = await model.explain(payload)
        else:
            model_handle = model
            response = await model_handle.remote(payload, model_type=ModelType.EXPLAINER)
        return response

    async def infer(
        self, payload, name: str, version: str = None
    ):
        model = self.get_model_from_registry(name, version)
        prediction = (await model.predict(payload)) if inspect.iscoroutinefunction(model.predict) \
            else model.predict(payload)
        return prediction

    async def list(self):
        return {"models": list(self._model_registry.get_models().keys())}

    async def ready(self):
        models = self._model_registry.get_models().values()
        is_ready = all([model.ready for model in models])
        return {"ready": is_ready}

    async def status(self, name: str, version: str = None):
        self.get_model_from_registry(name, version)
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

    def get_model_from_registry(self, name: str, version: str = None):
        model = self._model_registry.get_model(name, version)
        if model is None:
            raise tornado.web.HTTPError(
                status_code=404,
                reason="Model with name %s does not exist." % name if not version
                else "Model with name %s and version %s does not exist." % (name, version)
            )
        return model
