#!/usr/bin/env python3
"""Simple Flask example to serve Keras model."""

import glob
import json
import os

import numpy as np

from joblib import load
from pathlib import Path

from flask import Flask
from flask import abort
from flask import request

from flask_restplus import fields
from flask_restplus import Resource, Api, Namespace

from http import HTTPStatus

import tensorflow as tf

from sklearn.preprocessing.label import LabelEncoder
from tensorflow import keras

from demo_lib import demo


_HERE = Path(__file__).parent
_EXAMPLE_DATA_PATH = Path(_HERE, 'data/dance/floss/example.json')

_MODEL_DIR = Path(os.getenv("MODEL_DIR", _HERE / "models/v4"))
"""Path to model directory."""
_MODEL: keras.Model = None
"""Keras model."""

_ENCODER: LabelEncoder = None
"""Label encoder/decoder."""


app = Flask(__name__)

probe_ns = Namespace('probe', description="Health checks.")
model_ns = Namespace('model', description="Model namespace.")


example = json.loads(
    Path(_EXAMPLE_DATA_PATH).read_text())

model_input = model_ns.model(u'ModelInput', {
    'gesture': fields.String(
        require=False,
        description="String. Label of the dance move.",
        example=example['gesture'],
    ),
    'motion': fields.List(
        fields.List(fields.Float),
        required=True,
        description="Array of floats. Model input motion data.",
        example=example['motion']
    ),
    'orientation': fields.List(
        fields.List(fields.Float),
        required=True,
        description="Array of floats. Model input orientation data.",
        example=example['orientation']
    ),
    "playerId": fields.String(
        required=False,
        description="String. Player ID.",
        example=example['playerId']
    ),
    "type": fields.String(
        required=False,
        description="String. Type of data.",
        example=example['type']
    ),
    "uuid": fields.String(
        required=False,
        description="String. UUID",
        example=example['uuid']
    )
})
model_output = model_ns.model(u'ModelOutput', {
    'candidate': fields.String(
        required=True,
        description="Best matching candidate."
    ),
    'candidate_score': fields.Float(
        required=True,
        description="Best matching candidate's score."
    ),
    'predictions': fields.Raw(
        required=True,
        description="Mapping candidate -> score for each predicted candidate."
    ),
})

payload = model_ns.model('Payload', {
    'payload': fields.Nested(
        model_output,
        as_list=True,
        required=True,
        description="Array of model predictions for each input."
    ),
    'total': fields.Integer(
        required=False,
        description="Total number of predictions.")
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

    @model_ns.response(200, 'Success', model=payload)
    @model_ns.response(400, 'Validation Error')
    @model_ns.expect(model_input, validate=True)
    @model_ns.marshal_list_with(payload)
    def post(self):
        """Return predictions from the trained model.

        Expected input: tensor of shape (13,)
        """
        global _SESSION

        model, encoder = _load_keras_model()

        message = request.get_json(force=True)

        data = [demo.process_motion_rotation(message, 0)]
        instances = demo.create_dataframe(data, False)

        input_t: np.ndarray = np.array(instances, dtype=np.float64)

        tf.keras.backend.set_session(_SESSION)

        scores: np.ndarray = model.predict(
            input_t, max_queue_size=20, use_multiprocessing=True, workers=4)
        labels: np.ndarray = np.argmax(scores, axis=1)

        candidates: list = encoder.inverse_transform(labels).tolist()

        response = {
            'payload': [
                {
                    'candidate': candidates[i],
                    'candidate_score': float(sample[labels[i]]),
                    'predictions': dict(
                        zip(encoder.classes_, sample.tolist())),
                } for i, sample in enumerate(scores)
            ],
            'total': len(scores)
        }

        return response, HTTPStatus.OK


def _load_keras_model():
    """Load Keras model."""
    global _MODEL
    global _ENCODER
    global _SESSION

    if all([_MODEL, _ENCODER]):
        return _MODEL, _ENCODER

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
        _MODEL = keras.models.load_model(latest, custom_objects={
            'log_softmax': tf.nn.log_softmax,
            'softmax_cross_entropy_with_logits_v2_helper': tf.nn.softmax_cross_entropy_with_logits_v2
        })
        _ENCODER = load(_MODEL_DIR / 'encoder.joblib')

        if isinstance(_MODEL, keras.Model):
            app.logger.info(f"Model '{latest}' successfully loaded.")

        else:
            msg = f"Expected model of type: {keras.Model}, got {type(_MODEL)}"
            app.logger.error(msg)

            abort(HTTPStatus.BAD_REQUEST, "Failed. Model not loaded.")

        if isinstance(_ENCODER, LabelEncoder):
            app.logger.info(f"Encoder successfully loaded.")

        else:
            msg = f"Expected model of type: {LabelEncoder}, got {type(_ENCODER)}"
            app.logger.error(msg)

            abort(HTTPStatus.BAD_REQUEST, "Failed. Encoder not loaded.")

        _SESSION = tf.keras.backend.get_session()

    return _MODEL, _ENCODER


if __name__ == '__main__':

    api = Api(title="Gestures model serving")
    api.init_app(app)

    api.add_namespace(probe_ns)
    api.add_namespace(model_ns)

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
