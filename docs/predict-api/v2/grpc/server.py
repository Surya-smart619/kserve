from servicer import InferenceServicer
from kserve.handlers.dataplane import DataPlane
import grpc
from concurrent import futures
import grpc_predict_v2_pb2
import grpc_predict_v2_pb2_grpc

class GRPCServer:
    def __init__(
        self,
        data_plane: DataPlane
    ):
        self._data_plane = data_plane

    def _create_server(self):
        self._inference_servicer = InferenceServicer(self._data_plane)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        grpc_predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(self._server)
        self._server.add_insecure_port('[::]:50051')
        return self._server

    async def start(self):
        self._create_server()

        await self._server.start()

        print(
            "gRPC server running on "
            f"[::]:{50051}"
        )
        await self._server.wait_for_termination()

    async def stop(self, sig: int = None):
        # logger.info("Waiting for gRPC server shutdown")
        # TODO: Read from config
        await self._server.stop(grace=5)
        # logger.info("gRPC server shutdown complete")