import argparse
import base64
import json
import logging
import os

from google.protobuf.json_format import MessageToDict

import grpc
import grpc_predict_v2_pb2 as pb
import grpc_predict_v2_pb2_grpc


def run(host, port, hostname, model, endpoint, input_path):
    # os.environ.pop('HTTP_PROXY', None)
    if hostname:
        host_option = (('grpc.ssl_target_name_override', hostname,), ('grpc.enable_http_proxy', 0,),)
    else:
        host_option = None
    with grpc.insecure_channel(f'{host}:{port}', options=host_option) as channel:
    # with grpc.insecure_channel('10.97.76.113', options=host_option) as channel:
    # with grpc.secure_channel(
    #     f'{host}:{port}', grpc.ssl_channel_credentials(),
    #     options=(('grpc.ssl_target_name_override', hostname,),)
    # ) as channel:
        stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
        response = None
        if endpoint == 'server_metadata':
            response = stub.ServerMetadata(pb.ServerMetadataRequest())
        if endpoint == 'server_live':
            response = stub.ServerLive(pb.ServerLiveRequest())
        elif endpoint == 'server_ready':
            response = stub.ServerReady(pb.ServerReadyRequest())
        elif endpoint == 'model_ready':
            response = stub.ModelReady(pb.ModelReadyRequest(name=model))
        elif endpoint == 'model_metadata':
            response = stub.ModelMetadata(pb.ModelMetadataRequest(name=model))
        elif endpoint == 'infer' or endpoint == 'predict':
            with open(input_path) as json_file:
                data = json.load(json_file)
            # for (i, datum) in enumerate(data):
            #     if (datum["datatype"] == "BASE64"):
            #         if (datum["contents"]["bytes_contents"]):
            #             datum["contents"]["bytes_contents"][0] = base64.b64decode(datum["contents"]["bytes_contents"][0])



            # ======================Below are all gRPC custom model example=========================
            payload = [
                {
                    "name": "input-0",
                    "shape": [],
                    "datatype": "BYTES",
                    "contents": {
                        "bytes_contents": [base64.b64decode(data["inputs"][0]["data"][0])]
                    }
                }
            ]
            # ======================================================================================
            # response = MessageToDict(pb.ModelInferRequest(model_name=model, inputs=payload))
            response = stub.ModelInfer(pb.ModelInferRequest(model_name=model, inputs=payload), metadata=(('request_type', 'grpc_v2'), ('response_type', 'grpc_v2')))
        elif endpoint == 'unload':
            response = stub.RepositoryModelUnload(pb.RepositoryModelUnloadRequest(model_name=model))
            pass
        elif endpoint == 'load':
            response = stub.RepositoryModelLoad(pb.RepositoryModelLoadRequest(model_name=model))
            pass

        print("Response >>>>>>>>>>>", endpoint, ">>>>", response)
        print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTtt', type(response))

if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Ingress Host Name', default='localhost', type=str)
    parser.add_argument('--port', help='Ingress Port', default=80, type=int)
    parser.add_argument('--model', help='TensorFlow Model Name', type=str)
    parser.add_argument('--endpoint', help='Endpoint of model', default='model_metadata', type=str)
    parser.add_argument('--hostname', help='Service Host Name', default='', type=str)
    parser.add_argument('--input_path', help='Prediction data input path',
                        default='./input.json', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.hostname, args.model, args.endpoint, args.input_path)