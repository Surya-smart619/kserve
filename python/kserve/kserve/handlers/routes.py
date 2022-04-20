import json
import logging
import re
from kserve.handlers import DataPlane
from kserve.handlers.base import BaseHandler


class Routes(BaseHandler):
    def initialize(self, dataplane: DataPlane):
        self.dataplane = dataplane

    async def get(self, name: str):
        response = None
        if re.compile('/v2/models/([a-zA-Z0-9_-]+)/model-metadata').match(self.request.uri):
            response = await self.dataplane.model_metadata(name)
        #TODO: list v2 get apis
        self.write(response)

    async def post(self, name: str):
        logging.info("Requested url %s", self.request.uri)
        body = json.loads(self.request.body)
        response = None
        if re.compile('/v2/models/([a-zA-Z0-9_-]+)/infer').match(self.request.uri):
            response = await self.dataplane.infer(body, name)
        if re.compile('/v2/models/([a-zA-Z0-9_-]+)/explain').match(self.request.uri):
            response = await self.dataplane.explain(body, name)
        #TODO: list v2 post apis
        self.write(response)
