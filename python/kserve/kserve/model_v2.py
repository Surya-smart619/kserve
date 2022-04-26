import inspect
import json
import sys
from http import HTTPStatus
from typing import Dict, Union

import tornado.web
from cloudevents.http import CloudEvent
from tornado.httpclient import AsyncHTTPClient
from tritonclient.grpc import InferResult
from tritonclient.grpc.service_pb2 import ModelInferRequest, ModelInferResponse

from kserve.utils.utils import is_structured_cloudevent

EXPLAINER_FORMAT = "http://{0}/v2/models/{1}/explain"
PREDICTOR_URL_FORMAT = "http://{0}/v2/models/{1}/infer"


class ModelV2:

    def __init__(self, name: str):
        self.name = name
        self.ready = False
        self.predictor_host = None
        self.explainer_host = None
        self.timeout = 600
        self._http_client_instance = None

    @property
    def _http_client(self):
        if self._http_client_instance is None:
            self._http_client_instance = AsyncHTTPClient(max_clients=sys.maxsize)
        return self._http_client_instance

    def load(self) -> bool:
        self.ready = True
        return self.ready

    async def preprocess(self, request: Union[Dict, CloudEvent]) -> Union[Dict, ModelInferRequest]:
        """
        The preprocess handler can be overridden for data or feature transformation.
        The default implementation decodes to Dict if it is a binary CloudEvent
        or gets the data field from a structured CloudEvent.
        :param request: Dict|CloudEvent|ModelInferRequest
        :return: Transformed Dict|ModelInferRequest which passes to predict handler
        """
        response = request

        if isinstance(request, CloudEvent):
            response = request.data
            # Try to decode and parse JSON UTF-8 if possible, otherwise
            # just pass the CloudEvent data on to the predict function.
            # This is for the cases that CloudEvent encoding is protobuf, avro etc.
            try:
                response = json.loads(response.decode('UTF-8'))
            except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
                # If decoding or parsing failed, check if it was supposed to be JSON UTF-8
                if "content-type" in request._attributes and \
                        (request._attributes["content-type"] == "application/cloudevents+json" or
                         request._attributes["content-type"] == "application/json"):
                    raise tornado.web.HTTPError(
                        status_code=HTTPStatus.BAD_REQUEST,
                        reason=f"Failed to decode or parse binary json cloudevent: {e}"
                    )

        elif isinstance(request, dict):
            if is_structured_cloudevent(request):
                response = request["data"]

        return response

    def postprocess(self, response: Union[Dict, ModelInferResponse]) -> Dict:
        """
        The postprocess handler can be overridden for inference response transformation
        :param response: Dict|ModelInferResponse passed from predict handler
        :return: Dict
        """
        if isinstance(response, ModelInferResponse):
            response = InferResult(response)
            return response.get_response(as_json=True)
        return response

    async def _http_predict(self, request: Dict) -> Dict:
        request = await self.preprocess(request) if inspect.iscoroutinefunction(self.preprocess) \
            else self.preprocess(request)
        request = self.validate(request)
        predict_url = PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name)
        json_header = {'Content-Type': 'application/json'}
        http_response = await self._http_client.fetch(
            predict_url,
            method='POST',
            request_timeout=self.timeout,
            headers=json_header,
            body=json.dumps(request)
        )
        if http_response.code != 200:
            raise tornado.web.HTTPError(
                status_code=http_response.code,
                reason=http_response.body)
        response = json.loads(http_response.body)
        response = self.postprocess(response) if inspect.iscoroutinefunction(self.postprocess) \
            else self.postprocess(request)
        return response

    def validate(self, request):
        if "inputs" in request and not isinstance(request["inputs"], list):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Expected \"inputs\" to be a list"
            )
        return request

    async def predict(self, request: Union[Dict, ModelInferRequest]) -> Union[Dict, ModelInferResponse]:
        if not self.predictor_host:
            raise NotImplementedError
        return await self._http_predict(request)

    async def metadata(self):
        return {
            'name': self.name,
            'is_ready': self.ready,
            'predictor_host': self.predictor_host,
            'explainer_host': self.explainer_host
        }

    async def explain(self, request: Dict) -> Dict:
        if self.explainer_host is None:
            raise NotImplementedError
        request = await self.preprocess(request) if inspect.iscoroutinefunction(self.preprocess) \
            else self.preprocess(request)
        request = self.validate(request)
        explain_url = EXPLAINER_FORMAT.format(self.explainer_host, self.name)
        http_response = await self._http_client.fetch(
            url=explain_url,
            method='POST',
            request_timeout=self.timeout,
            body=json.dumps(request)
        )
        if http_response.code != 200:
            raise tornado.web.HTTPError(
                status_code=http_response.code,
                reason=http_response.body)
        response = json.loads(response.body)
        response = self.postprocess(response) if inspect.iscoroutinefunction(self.postprocess) \
            else self.postprocess(request)
        return response
