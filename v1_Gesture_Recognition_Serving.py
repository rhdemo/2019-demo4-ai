#!/usr/bin/env python3
"""Simple Flask example to serve Keras model."""

import glob
import json
import os

import numpy as np

from pathlib import Path

from flask import Flask
from flask import abort
from flask import request

from flask_restplus import fields
from flask_restplus import Resource, Api, Namespace

from http import HTTPStatus

import tensorflow as tf

from tensorflow import keras


_HERE = Path(__file__).parent
_ACCELEROMETER_DATA_PATH = Path(_HERE, 'data/accelerometer/xo')

_MODEL_DIR = Path(os.getenv("MODEL_DIR", _HERE / "models/hdf5"))
"""Path to model directory."""
_MODEL: keras.Model = None
"""Keras model."""
_GRAPH: tf.Graph = None
"""Default TensorFlow graph."""


app = Flask(__name__)

probe_ns = Namespace('probe', description="Health checks.")
model_ns = Namespace('model', description="Model namespace.")

model_input = model_ns.model('Input', {
    'instances': fields.List(
        fields.List(fields.Float),
        required=True,
        description="Model input instances. Tensor of shape (N, 13)",
        example=json.loads(
            Path(_ACCELEROMETER_DATA_PATH, 'examples/example_instance.json').read_text()),
    ),
    'signature': fields.String(
        required=True,
        default="serving_default",
        description="Signature to be returned my model.",
        example="serving_default")
})

model_ns.add_model('model_input', model_input)


@probe_ns.route('/liveness')
class Liveness(Resource):

    # noinspection PyMethodMayBeStatic
    def get(self):
        """Heartbeat."""
        return {'Status': "Running OK."}, HTTPStatus.OK


@probe_ns.route('/readiness')
class Readiness(Resource):

    # noinspection PyMethodMayBeStatic
    def get(self):
        """Readiness."""
        if _MODEL is not None:
            response = {'Status': "Ready."}, HTTPStatus.OK
        else:
            response = {'Status': "Model has not been loaded."}, \
                       HTTPStatus.SERVICE_UNAVAILABLE

        return response


@model_ns.route('/predict')
class Model(Resource):
    """Model api resource."""

    @model_ns.expect(model_input, validate=True)
    def post(self):
        """Return predictions from the trained model.

        Expected input: tensor of shape (13,)
        """
        message = request.get_json(force=True)
        input_t: np.ndarray = np.array(message['instances'], dtype=np.float64)

        with _GRAPH.as_default():
            predictions: np.ndarray = _MODEL.predict_on_batch(input_t)

        response = {
            'predictions': predictions.tolist()
        }

        return response, HTTPStatus.OK


@app.before_first_request
def _load_keras_model():
    """Load Keras model."""
    global _MODEL
    global _GRAPH

    app.logger.info("Loading Keras model.")

    # TODO: glob by pattern according to our model file naming
    model_files = sorted(
        # This serving is made for .h5 models
        glob.iglob(str(_MODEL_DIR / '**/*.h5'), recursive=True), key=os.path.getctime)

    if not model_files:
        msg = f"Empty directory provided: {_MODEL_DIR}."
        app.logger.error(msg)

        # TODO: maybe BAD_REQUEST is not ideal here
        abort(HTTPStatus.BAD_REQUEST, "Failed. Model not found.")

    else:
        latest = model_files[-1]
        # set the global
        _MODEL = keras.models.load_model(latest)

        if isinstance(_MODEL, keras.Model):
            app.logger.info(f"Model '{latest}' successfully loaded.")

        else:
            msg = f"Expected model of type: {keras.Model}, got {type(_MODEL)}"
            app.logger.error(msg)

            abort(HTTPStatus.BAD_REQUEST, "Failed. Model not loaded.")

        _GRAPH = tf.get_default_graph()


if __name__ == '__main__':

    api = Api(title="Gestures model serving")
    api.init_app(app)

    api.add_namespace(probe_ns)
    api.add_namespace(model_ns)

    app.run(host='0.0.0.0', debug=True)
