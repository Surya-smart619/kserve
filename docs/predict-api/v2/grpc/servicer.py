import grpc_predict_v2_pb2 as predict_pb
import grpc_predict_v2_pb2_grpc
from kserve.handlers.dataplane import DataPlane

class InferenceServicer(grpc_predict_v2_pb2_grpc.GRPCInferenceServiceServicer):

    def __init__(
        self, data_plane: DataPlane
    ):
        super().__init__()
        self._data_plane = data_plane

    async def ServerLive(self, request, context):
        is_live = await self._data_plane.live()
        return predict_pb.ServerLiveResponse(live=is_live)


    async def ServerReady(self, request, context):
        is_ready = await self._data_plane.ready()
        return predict_pb.ServerReadyResponse(ready=is_ready)

