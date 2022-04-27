from enum import Enum
import json
import logging
import re
from kserve.handlers import DataPlane
from kserve.handlers.base import BaseHandler


class Enpoints(Enum):
    # List models
    LIST_MODEL = re.compile(r"/models")

    # Server metadata
    METADATA = re.compile(r"/")

    # Liveness and readiness
    LIVE = re.compile(r"/health/live")
    READY = re.compile(r"/health/ready")

    # Model metadata
    MODEL_METADATA = re.compile(
        r"/models/([a-zA-Z0-9_-]+)|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)")

    # Model ready
    MODEL_STATUS = re.compile(
        r"/models/([a-zA-Z0-9_-]+)/ready|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)/ready")

    # Model infer
    INFER = re.compile(
        r"/models/([a-zA-Z0-9_-]+)/infer|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)/infer")

    # Model Explain
    EXPLAIN = re.compile(
        r"/models/([a-zA-Z0-9_-]+)/explain|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)/explain")

    # Model Repository API
    LOAD = re.compile(
        r"/models/([a-zA-Z0-9_-]+)/load|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)/load")
    UNLOAD = re.compile(
        r"/models/([a-zA-Z0-9_-]+)/unload|/models/([a-zA-Z0-9_-]+)/versions/([.a-zA-Z0-9_-]+)/unload")


class PathParams(Enum):
    MODEL_NAME = re.compile(r"/models/(?P<model_name>[^\/]+)(.*)")
    MODEL_NAME_AND_VERSION = re.compile(
        r"/models/(?P<model_name>[^\/]+)/versions/(?P<model_version>[^\/]+)(.*)")


class Routes(BaseHandler):
    def initialize(self, dataplane: DataPlane):
        self.dataplane = dataplane

    def get_path_params(self, request_url):
        params = {}
        matched_patterns = PathParams.MODEL_NAME_AND_VERSION.value.fullmatch(request_url) or \
            PathParams.MODEL_NAME.value.fullmatch(request_url)
        if matched_patterns:
            params = matched_patterns.groupdict()
        logging.info('Path params %s', params)
        return params

    async def get(self, request_url="/"):
        logging.info('Requested Url ::%s', request_url)
        params = self.get_path_params(request_url)
        response = None
        if Enpoints.METADATA.value.fullmatch(request_url):
            response = await self.dataplane.metadata()
        elif Enpoints.LIST_MODEL.value.fullmatch(request_url):
            response = await self.dataplane.list()
        elif Enpoints.LIVE.value.fullmatch(request_url):
            response = await self.dataplane.live()
        elif Enpoints.READY.value.fullmatch(request_url):
            response = await self.dataplane.ready()
        elif Enpoints.MODEL_METADATA.value.fullmatch(request_url):
            response = await self.dataplane.model_metadata(params["model_name"], params.get("model_version"))
        elif Enpoints.MODEL_STATUS.value.fullmatch(request_url):
            response = await self.dataplane.status(params["model_name"], params.get("model_version"))
        self.write(response)

    async def post(self, request_url):
        logging.info("Requested url %s", request_url)
        body = json.loads(self.request.body)
        params = self.get_path_params(request_url)
        response = None
        if Enpoints.INFER.value.fullmatch(request_url):
            response = await self.dataplane.infer(body, params["model_name"], params.get("model_version"))
        elif Enpoints.EXPLAIN.value.fullmatch(request_url):
            response = await self.dataplane.explain(body, params["model_name"], params.get("model_version"))
        elif Enpoints.LOAD.value.fullmatch(request_url):
            response = await self.dataplane.load(params["model_name"], params.get("model_version"))
        elif Enpoints.UNLOAD.value.fullmatch(request_url):
            response = await self.dataplane.unload(params["model_name"], params.get("model_version"))
        self.write(response)
