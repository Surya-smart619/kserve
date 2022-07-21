import logging
from statistics import mode
from kserve_v2.grpc import grpc_predict_v2_pb2 as pb
from kserve_v2.grpc import grpc_predict_v2_pb2_grpc
from kserve_v2.handlers.dataplane import DataPlane


class InferenceServicer(grpc_predict_v2_pb2_grpc.GRPCInferenceServiceServicer):

    def __init__(
        self, data_plane: DataPlane
    ):
        super().__init__()
        self._data_plane = data_plane

    async def ServerLive(
        self, request: pb.ServerLiveRequest, context
    ) -> pb.ServerLiveResponse:
        is_live = await self._data_plane.live()
        return pb.ServerLiveResponse(live=is_live)

    async def ServerReady(
        self, request: pb.ServerReadyRequest, context
    ) -> pb.ServerLiveResponse:
        is_ready = await self._data_plane.ready()
        return pb.ServerReadyResponse(ready=is_ready)

    async def ModelReady(
        self, request: pb.ModelReadyRequest, context
    ) -> pb.ModelMetadataResponse:
        is_ready = await self._data_plane.model_ready(model_name=request.name, model_version=request.version)
        return pb.ModelReadyResponse(ready=is_ready)

    async def ModelMetadata(
        self, request: pb.ModelMetadataRequest, context
    ) -> pb.ModelMetadataResponse:
        metadata = await self._data_plane.model_metadata(model_name=request.name, model_version=request.version)
        # TODO: Use structured class and convert
        return pb.ModelMetadataResponse(name=metadata["model_name"], versions=metadata["model_versions"])

    async def RepositoryModelLoad(
        self, request: pb.RepositoryModelLoadRequest, context
    ) -> pb.RepositoryModelLoadResponse:
        response = await self._data_plane.load(name=request.model_name)
        return pb.RepositoryModelLoadResponse(model_name=response["name"], isLoaded=response["load"])

    async def RepositoryModelUnload(
        self, request: pb.RepositoryModelUnloadRequest, context
    ) -> pb.RepositoryModelUnloadResponse:
        response = await self._data_plane.unload(name=request.model_name)
        return pb.RepositoryModelUnloadResponse(model_name=response["name"], isUnloaded=response["unload"])

    async def ModelInfer(
        self, request: pb.ModelInferRequest, context
    ) -> pb.ModelInferResponse:
        # TODO: Use structured class and convert
        # NOTE: All the below convertion are to test infer
        fields = request.inputs[0].contents.ListFields()
        _, field_value = fields[0]
        points = list(field_value)
        payload = {
            "instances": [points[:4], points[4:]]
        }
        response = await self._data_plane.infer(payload=payload, model_name=request.model_name, model_version=request.model_version)
        output = response["output"]
        return pb.ModelInferResponse(
            model_name=response["model_name"],
            model_version=response["model_version"],
            id=response["id"],
            outputs=[
                {
                    "name": output["name"],
                    "shape": output["shape"],
                    "datatype": output["datatype"],
                    "contents": {
                        "int_contents": output["data"]
                    }
                }

            ]
        )
