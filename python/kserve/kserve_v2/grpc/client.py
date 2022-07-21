import logging

import grpc
import grpc_predict_v2_pb2 as pb
import grpc_predict_v2_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
        response = stub.ServerLive(pb.ServerLiveRequest())
        print('ServerLiveRequest>>>>>>', response)
        response = stub.ServerReady(pb.ServerReadyRequest())
        print('ServerReadyRequest>>>>>>', response)
        response = stub.ModelReady(pb.ModelReadyRequest(name='sklearn-irisv2'))
        print('ModelReadyRequest>>>>>>', response)
        response = stub.ModelMetadata(pb.ModelMetadataRequest(name='sklearn-irisv2'))
        print('ModelReadyRequest>>>>>>', response)
        inputs = [
            {
                "name": "input-0",
                "shape": [2, 4],
                "datatype": "FP32",
                "contents": {
                    "fp32_contents": [6.8, 2.8, 4.8, 1.4, 6.0, 3.4, 4.5, 1.6]
                }
            }
        ]

        response = stub.ModelInfer(pb.ModelInferRequest(model_name='sklearn-irisv2', inputs=inputs))
        print('ModelInferRequest>>>>>>', response)
        response = stub.RepositoryModelUnload(pb.RepositoryModelUnloadRequest(model_name='sklearn-irisv2'))
        print('ModelUnLoad>>>>>>>>>>', response)
        response = stub.RepositoryModelLoad(pb.RepositoryModelLoadRequest(model_name='sklearn-irisv2'))
        print('ModelLoad>>>>>>>>>>>>', response)

if __name__ == '__main__':
    logging.basicConfig()
    run()
