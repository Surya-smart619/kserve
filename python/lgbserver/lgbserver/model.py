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

import kserve
import lightgbm as lgb
from lightgbm import Booster
import os
from typing import Dict
import pandas as pd
from kserve.errors import InferenceError, ModelMissingError

from kserve.protocol.infer_type import InferOutput, InferRequest, InferResponse

from kserve.utils.utils import generate_uuid

MODEL_EXTENSIONS = (".bst")


class LightGBMModel(kserve.Model):
    def __init__(self, name: str, model_dir: str, nthread: int,
                 booster: Booster = None):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.nthread = nthread
        if booster is not None:
            self._booster = booster
            self.ready = True

    def load(self) -> bool:
        model_path = kserve.Storage.download(self.model_dir)
        model_files = []
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path) and file.endswith(MODEL_EXTENSIONS):
                model_files.append(file_path)
        if len(model_files) == 0:
            raise ModelMissingError(model_path)
        elif len(model_files) > 1:
            raise RuntimeError('More than one model file is detected, '
                               f'Only one is allowed within model_dir: {model_files}')
        self._booster = lgb.Booster(params={"nthread": self.nthread},
                                    model_file=model_files[0])
        self.ready = True
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            if isinstance(payload, Dict):
                dfs = []
                for input in payload['inputs']:
                    dfs.append(pd.DataFrame(input, columns=self._booster.feature_name()))
                inputs = pd.concat(dfs, axis=0)

                result = self._booster.predict(inputs)
                return {"predictions": result.tolist()}

            elif isinstance(payload, InferRequest):
                dfs = []
                infer_input = payload.to_rest()
                instances = infer_input["inputs"][0]["data"]
                for input in instances:
                    dfs.append(pd.DataFrame(input, columns=self._booster.feature_name()))
                inputs = pd.concat(dfs, axis=0)
                result = self._booster.predict(inputs)
                response_id = generate_uuid()
                infer_output = InferOutput(name="output-0", shape=list(infer_input["inputs"][0]["shape"]), datatype=infer_input["inputs"][0]["datatype"], data=result.tolist())
                infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], request_id=response_id)
                return infer_response
                
        except Exception as e:
            raise InferenceError(str(e))
