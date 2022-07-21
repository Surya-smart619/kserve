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

import argparse
import asyncio
import logging

import kserve_v2
from sklearnserver import SKLearnModel, SKLearnModelRepository
from kserve_v2.model import ModelMissingError

DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/tmp/model"
DEFAULT_MODEL_VERSION = "1.1.1"
parser = argparse.ArgumentParser(parents=[kserve_v2.model_server.parser])
parser.add_argument('--model_dir', required=True,
                    help='A URI pointer to the model binary')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--model_version', default=DEFAULT_MODEL_VERSION,
                    help='The version that the model is served under.')
# TODO: Suports dynamic version in arg
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = SKLearnModel(args.model_name, args.model_dir, args.model_version)
    try:
        model.load()

    except ModelMissingError:
        logging.error(f"fail to locate model file for model {args.model_name} under dir {args.model_dir},"
                      f"trying loading from model repository.")

    asyncio.run(kserve_v2.ModelServer(registered_models=SKLearnModelRepository(
        args.model_dir)).start([model] if model.ready else []))
