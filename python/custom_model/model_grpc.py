# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
from typing import Dict, Union

import kserve
import torch
from kserve.grpc.grpc_predict_v2_pb2 import ModelInferRequest
from PIL import Image
from torchvision import models, transforms


class AlexNetModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        model = models.alexnet(pretrained=True)
        model.eval()
        self.model = model
        self.ready = True

    def predict(self, request: Union[Dict, ModelInferRequest]) -> Dict:
        inputs = request["instances"]
        print(">>>>>>>>>>>>>>>>>", inputs)
        input_image = Image.open(io.BytesIO(inputs[0]))

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        output = self.model(input_batch)

        torch.nn.functional.softmax(output, dim=1)[0]

        values, top_5 = torch.topk(output, 5)

        return {"predictions": values.tolist()}


if __name__ == "__main__":
    model = AlexNetModel("custom-model")
    model.load()
    asyncio.run(
        kserve.ModelServer(workers=1).start([model])
    )
