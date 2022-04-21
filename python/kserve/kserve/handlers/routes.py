import json
import logging
import re
from kserve.handlers import DataPlane
from kserve.handlers.base import BaseHandler


class Routes(BaseHandler):
    def initialize(self, dataplane: DataPlane):
        self.dataplane = dataplane

    async def get(self, name: str = None):
        # self.path_args
        logging.info('async def get(self, name: str):')
        request_uri = self.request.uri
        response = None
        if re.compile('/v2/models').fullmatch(request_uri):
            response = await self.dataplane.list()
        elif re.compile('/v2/health/live').fullmatch(request_uri):
            response = await self.dataplane.live()
        elif re.compile('/v2/models/([a-zA-Z0-9_-]+)/model-metadata').match(request_uri):
            response = await self.dataplane.model_metadata(name)
        elif re.compile('/v2/models/([a-zA-Z0-9_-]+)/status').match(request_uri):
            response = await self.dataplane.status(name)
        # TODO: list v2 get apis
        self.write(response)

    async def post(self, name: str):
        request_uri = self.request.uri
        logging.info("Requested url %s", request_uri)
        body = json.loads(self.request.body)
        response = None
        if re.compile('/v2/models/([a-zA-Z0-9_-]+)/infer').match(request_uri):
            response = await self.dataplane.infer(body, name)
        elif re.compile('/v2/models/([a-zA-Z0-9_-]+)/explain').match(request_uri):
            response = await self.dataplane.explain(body, name)
        elif re.compile('/v2/models/([a-zA-Z0-9_-]+)/load').match(request_uri):
            response = await self.dataplane.load(name)
        elif re.compile('/v2/models/([a-zA-Z0-9_-]+)/unload').match(request_uri):
            response = await self.dataplane.unload(name)
        # TODO: list v2 post apis
        self.write(response)
