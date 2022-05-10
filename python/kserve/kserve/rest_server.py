import argparse
import logging
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.routing import APIRoute as FastAPIRoute

import kserve.handlers as handlers
from kserve.model import Model
from kserve.model_repository import ModelRepository

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 8081
DEFAULT_MAX_BUFFER_SIZE = 104857600

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--http_port', default=DEFAULT_HTTP_PORT, type=int,
                    help='The HTTP Port listened to by the model server.')
parser.add_argument('--grpc_port', default=DEFAULT_GRPC_PORT, type=int,
                    help='The GRPC Port listened to by the model server.')
parser.add_argument('--max_buffer_size', default=DEFAULT_MAX_BUFFER_SIZE, type=int,
                    help='The max buffer size for tornado.')
parser.add_argument('--workers', default=1, type=int,
                    help='The number of works to fork')
parser.add_argument('--max_asyncio_workers', default=None, type=int,
                    help='Max number of asyncio workers to spawn')

args, _ = parser.parse_known_args()


class RestServer:
    def __init__(self, http_port: int = args.http_port,
                 grpc_port: int = args.grpc_port,
                 max_buffer_size: int = args.max_buffer_size,
                 workers: int = args.workers,
                 max_asyncio_workers: int = args.max_asyncio_workers,
                 registered_models: ModelRepository = ModelRepository()):
        self.registered_models = registered_models
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.max_buffer_size = max_buffer_size
        self.workers = workers
        self.max_asyncio_workers = max_asyncio_workers
        self._http_server = None

    def create_application(self):
        dataplane = handlers.DataPlane(model_registry=self.registered_models)
        routes = [
            # Server metadata
            FastAPIRoute(
                "/v2",
                dataplane.metadata,
            ),
            # Liveness and readiness
            FastAPIRoute(
                "/v2/health/live",
                dataplane.live,
            ),
            FastAPIRoute(
                "/v2/health/ready",
                dataplane.ready,
            ),
            # Model Ready
            FastAPIRoute(
                "/v2/models/{model_name}/ready",
                dataplane.model_ready,
            ),
            FastAPIRoute(
                "/v2/models/{model_name}/versions/{model_version}/ready",
                dataplane.model_ready,
            ),
            # Model metadata
            FastAPIRoute(
                "/v2/models/{model_name}",
                dataplane.model_metadata,
            ),
            FastAPIRoute(
                "/v2/models/{model_name}/versions/{model_version}",
                dataplane.model_metadata,
            ),
            # Model infer
            FastAPIRoute(
                "/v2/models/{model_name}/infer",
                dataplane.infer,
                methods=["POST"]
            ),
            FastAPIRoute(
                "/v2/models/{model_name}/versions/{model_version}/infer",
                dataplane.infer,
                methods=["POST"]
            ),
            # Model load and unload
            FastAPIRoute(
                "/v2/repository/models/{model_name}/load",
                dataplane.load,
                methods=["POST"],
            ),
            FastAPIRoute(
                "/v2/repository/models/{model_name}/unload",
                dataplane.unload,
                methods=["POST"],
            ),
        ]

        return FastAPI(
            routes=routes
        )

    async def start(self, models: List[Model], nest_asyncio: bool = False):
        if isinstance(models, list):
            for model in models:
                if isinstance(model, Model):
                    self.register_model(model)
                else:
                    raise RuntimeError("Model type should be Model")
        cfg = uvicorn.Config(
            self.create_application(),
            port=self.http_port,
        )
        self._server = uvicorn.Server(cfg)
        await self._server.serve()

    def register_model(self, model: Model):
        if not model.name:
            raise Exception(
                "Failed to register model, model.name must be provided.")
        self.registered_models.update(model)
        logging.info("Registering model: %s", model.name)
