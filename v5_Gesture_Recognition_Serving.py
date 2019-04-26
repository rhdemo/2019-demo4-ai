#!/usr/bin/env python3
"""Simple Flask example to serve Keras model."""

import glob
import json
import os

import logging

import numpy as np
import pandas as pd

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

from demo_lib.demo import process_motion_rotation
from demo_lib.demo import clean_data
from demo_lib.demo import featurize


_HERE = Path(__file__).parent
_EXAMPLE_DATA_PATH = Path(_HERE, 'data/dance/floss/example.json')

_MODEL_DIR = Path(os.getenv("MODEL_DIR", _HERE / "models/v5"))
"""Path to model directory."""
_MODEL: keras.Model = None
"""Keras model."""

_ENCODER: LabelEncoder = None
"""Label encoder/decoder."""


app = Flask(__name__)
app.logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

probe_ns = Namespace('probe', description="Health checks.")
model_ns = Namespace('model', description="Model namespace.")


example = json.loads(
    Path(_EXAMPLE_DATA_PATH).read_text())

model_input = model_ns.model(u'ModelInput', {
    'gesture': fields.String(
        require=True,
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
prediction_input = model_ns.model(u'PredictionInput', {
    'instances': fields.List(
        fields.Nested(model_input),
        required=True,
        description="List of inputs to the model for prediction."
    )
})

payload = model_ns.model('uPayload', {
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
    @model_ns.response(400, 'Validation Error or invalid model input')
    @model_ns.expect(prediction_input, validate=True)
    @model_ns.marshal_list_with(payload)
    def post(self):
        """Return predictions from the trained model."""
        global _SESSION

        model, encoder = _load_keras_model()

        bins = json.loads(
                Path(_MODEL_DIR / 'bins.json').read_text())
        bins_partial = json.loads(
                Path(_MODEL_DIR / 'bins_partial.json').read_text())

        instances = request.get_json(force=True)['instances']

        data = []
        for sample in instances:
            gesture = sample['gesture']
            data_clean = list(
                clean_data([process_motion_rotation(sample)]))

            if data_clean:

                _, feature_df_total = featurize(
                        data_clean, col_bins=bins[gesture])

                _, feature_df_partial = featurize(
                        data_clean, col_bins=bins_partial[gesture])

                data.append(np.vstack([feature_df_partial, feature_df_total]))

        input_t: np.ndarray = np.array(data, dtype=np.float64)

        if not np.any(input_t):
            return "Error: Invalid model input.", HTTPStatus.BAD_REQUEST 

        tf.keras.backend.set_session(_SESSION)

        scores: np.ndarray = model.predict(
            input_t, max_queue_size=20, use_multiprocessing=True, workers=4)
        probas: np.ndarray = np.exp(scores)
        probas /= np.sum(probas, axis=1)[:, np.newaxis]

        labels: np.ndarray = np.argmax(scores, axis=1)

        candidates: list = encoder.inverse_transform(labels).tolist()

        response = {
            'payload': [
                {
                    'candidate': candidates[i],
                    'candidate_score': float(sample[labels[i]]),
                    'predictions': dict(
                        zip(encoder.classes_, sample.tolist())),
                } for i, sample in enumerate(probas)
            ],
            'total': len(candidates)
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
        glob.iglob(str(_MODEL_DIR / '**/*.h5'), recursive=True), key=os.path.getctime, reverse=True)

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

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
