# Deploy Lightgbm model with InferenceService

## Creating your own model and testing the LightGBM server

To test the LightGBM Server, first we need to generate a simple LightGBM model using Python.

```python
import lightgbm as lgb
from sklearn.datasets import load_iris
import os

model_dir = "."
BST_FILE = "model.bst"

iris = load_iris()
y = iris['target']
X = iris['data']
dtrain = lgb.Dataset(X, label=y)

params = {
    'objective':'multiclass', 
    'metric':'softmax',
    'num_class': 3
}
lgb_model = lgb.train(params=params, train_set=dtrain)
model_file = os.path.join(model_dir, BST_FILE)
lgb_model.save_model(model_file)
```

Then, we can install and run the [LightGBM Server](https://github.com/kserve/kserve/python/lgbserver) using the generated model and test for prediction. Models can be on local filesystem, S3 compatible object storage, Azure Blob Storage, or Google Cloud Storage.

```shell
python -m lgbserver --model_dir /path/to/model_dir --model_name lgb
```

We can also do some simple predictions

```python
import requests

request = {'sepal_width_(cm)': {0: 3.5}, 'petal_length_(cm)': {0: 1.4}, 'petal_width_(cm)': {0: 0.2},'sepal_length_(cm)': {0: 5.1} }
formData = {
    'inputs': [request]
}
res = requests.post('http://localhost:8080/v1/models/lgb/infer', json=formData)
print(res)
print(res.text)
```

## Create the InferenceService

=== "Old Schema"

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "lightgbm-v2-iris"
spec:
  predictor:
    lightgbm:
      protocolVersion: v2
      storageUri: "gs://kfserving-examples/models/lightgbm/v2/iris"
```

=== "New Schema"

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "lightgbm-v2-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: lightgbm
      runtime: kserve-mlserver
      storageUri: "gs://kfserving-examples/models/lightgbm/v2/iris"
```

Apply the above yaml to create the InferenceService

```bash
kubectl apply -f lightgbm.yaml
```

==**Expected Output**==

```bash
inferenceservice.serving.kserve.io/lightgbm-v2-iris created
```

## Run a prediction

The first step is to [determine the ingress IP and ports](../../../get_started/first_isvc.md#3-determine-the-ingress-ip-and-ports) and set `INGRESS_HOST` and `INGRESS_PORT`

```text
MODEL_NAME=lightgbm-v2-iris
INPUT_PATH=@./iris-input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice lightgbm-v2-iris -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/$MODEL_NAME/infer -d $INPUT_PATH
```

==**Expected Output**==

```bash
* Trying 169.63.251.68...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8080 (#0)
> POST /v2/models/lightgbm-iris/infer HTTP/1.1
> Host: lightgbm-iris.default.example.com
> User-Agent: curl/7.58.0
> Accept: */*
> Content-Length: 213
> Content-Type: application/x-www-form-urlencoded
> 
* upload completely sent off: 213 out of 213 bytes
< HTTP/1.1 200 OK
< content-length: 85
< content-type: application/json; charset=UTF-8
< date: Wed, 11 May 2022 10:17:15 GMT
< server: istio-envoy
< x-envoy-upstream-service-time: 13
< 
* Connection #0 to host localhost left intact
{"predictions": [[0.9999378629898321, 4.415799218835629e-05, 1.797901797954119e-05]]}
```

## Run LightGBM InferenceService with your own image

Since the KServe LightGBM image is built from a specific version of `lightgbm` pip package, sometimes it might not be compatible with the pickled model
you saved from your training environment, however you can build your own lgbserver image following [this instruction](https://github.com/kserve/kserve/python/lgbserver/README.md#building-your-own-ligthgbm-server-docker-image).

To use your lgbserver image:

- Add the image to the KServe [configmap](https://github.com/kserve/kserve/config/configmap/inferenceservice.yaml)

```yaml
"lightgbm": {
    "image": "<your-dockerhub-id>/kserve/lgbserver",
},
```

- Specify the `runtimeVersion` on `InferenceService` spec

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "lightgbm-iris"
spec:
  predictor:
    lightgbm:
      storageUri: "gs://kfserving-examples/models/lightgbm/iris"
      runtimeVersion: X.X.X
```
