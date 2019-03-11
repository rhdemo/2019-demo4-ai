# SummitDemo

## Preprocessing

### Doodle data

Convert to images

### Accelerometer Data

Speed vs Distance profiles (2d)

Accelerometer time-series scaled to same time periods

## Models

Multi-class logistic regression

Multi-class SVMs

k-Nearest Neighbors

CNNs

RNNs on streaming data (for fun, if time)

---

## Serving


Two models have been exported:

- .hdf5 using keras default `save` method
- .pb using `tf.saved_model.simple_save` tensorflow api

Each exported model format requires different kind of treatment.

### .pb

#### First examine the model

See [this](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) for information about how to get the `saved_model_cli` tool.

```
saved_model_cli show --dir models/pb/* --all
```

You should see something similar to this in case of gestures model.

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_data'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 13)
        name: dense_1_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense_1/LogSoftmax:0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: dense_1/LogSoftmax:0
  Method name is: tensorflow/serving/predict
```

There are two key information we get from this: the *method name* and the *shape* of inputs and outputs.

<br>

#### To the serving part

```
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving
```

Specify path to the saved `*.pb` model directory.

NOTE: This is usually a top level directory containing multiple timestamps as exported by [the training jupyter notebook](v1_Gesture_Recognition_Training.ipynb).

```
# bash
SAVED_MODEL_DIR=`realpath models/pb/`
MODEL_NAME="gestures_v1"

# fish
set -x saved_model_dir (realpath models/pb/)
set -x MODEL_NAME gestures_v1
```

We wanna make use of the REST API which we will map 8501 port for.

```
# Start TensorFlow Serving container and open the REST API port
# bash
docker run --rm -p 8501:8501 \
    -v "$SAVED_MODEL_DIR:/models/$MODEL_NAME" \
    -e MODEL_NAME=$MODEL_NAME \
    -t tensorflow/serving

# fish
docker run --rm -p 8501:8501 \
  --mount type=bind,source=$SAVED_MODEL_DIR,target=/models/$MODEL_NAME \
  -e MODEL_NAME=$MODEL_NAME -t tensorflow/serving
```

<br>

```
# example input
# bash
example_fpath="data/accelerometer/xo/examples/example_instance.json"
example_instance="[`cat $example_fpath`]"

# fish
set -x example_fpath 'data/accelerometer/xo/examples/example_instance.json'
set -x example_instance '['(cat $example_fpath)']'
```

```
# Query the model using the predict API
# bash
curl -d "{\"instances\": $example_instance}" \
     -X POST http://localhost:8501/v1/models/$MODEL_NAME:predict

# fish
curl -d '{"instances": '$example_instance' }' \
	 -X POST 'http://localhost:8501/v1/models/'$MODEL_NAME':predict'

# Returns => { "predictions": [2.5, 3.0, 4.5] }
```

### hdf5

Custom flask server and client is requried to serve this model.

To run the server, [flask app is included](/home/macermak/code/SummitDemo/v1_Gesture_Recognition_Serving.py):

```bash
./*Serving.py
```


## Client

There is a `v1_Gesture_Recognition_Client.py` file which defines a simple client cli.

For more info:

```bash
./*Client.py --help
```

Example usage:
```bash
./*Client.py $example_fpath --model_name='gestures_v1'
```
