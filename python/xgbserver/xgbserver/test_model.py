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

import xgboost as xgb
import os
from sklearn.datasets import load_iris
from kserve.protocol.infer_type import InferInput, InferRequest
from xgbserver import XGBoostModel

model_dir = os.path.join(os.path.dirname(__file__), "example_model", "model")
BST_FILE = "model.bst"
NTHREAD = 1


def test_model():
    iris = load_iris()
    y = iris['target']
    X = iris['data']
    dtrain = xgb.DMatrix(X, label=y)
    param = {'max_depth': 6,
             'eta': 0.1,
             'silent': 1,
             'nthread': 4,
             'num_class': 10,
             'objective': 'multi:softmax'}
    xgb_model = xgb.train(params=param, dtrain=dtrain)
    model_file = os.path.join(model_dir, BST_FILE)
    xgb_model.save_model(model_file)
    model = XGBoostModel("model", model_dir, NTHREAD)
    model.load()
    request = [X[0].tolist()]
    response = model.predict({"instances": request})
    assert response["predictions"] == [0]


def test_model_v2():
    model = XGBoostModel("model", model_dir, NTHREAD)
    model.load()

    # Test input data structure for v2 
    request = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [2, 4],
                "datatype": "FP32",
                "data": [
                    [6.8, 2.8, 4.8, 1.4],
                    [6.0, 3.4, 4.5, 1.6]
                ]
            }
        ]
    }
    infer_input = InferInput(request["inputs"][0]["name"], request["inputs"][0]
                         ["shape"], request["inputs"][0]["datatype"], request["inputs"][0]["data"])
    infer_request = InferRequest(model_name = "xgboost", infer_inputs = [infer_input])
    infer_response = model.predict(infer_request)
    response = infer_response.to_rest()
    assert response["outputs"][0]["data"] == [1.0, 1.0]
