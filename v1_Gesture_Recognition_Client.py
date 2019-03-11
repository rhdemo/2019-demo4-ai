#!/usr/bin/env python3
"""Simple client for Keras model inference using tf-serving."""

import sys
import json

import click
import requests

from pathlib import Path
from typing import List


_DEFAULT_DATA_MODEL = {
    'signature_name': 'serving_default',
    'instances': []
}
"""Default JSON model."""

DEFAULT_HOST_URL = "http://localhost"
"""Default server url."""
DEFAULT_HOST_PORT = 5000
"""Default port."""

DEFAULT_MODEL_DIR = 'models'
DEFAULT_MODEL_NAME = 'model'


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    'input_file', type=click.Path(exists=True),
)
@click.option(
    '--signature', type=str,
    default=_DEFAULT_DATA_MODEL['signature_name'],
    help="Signature to be returned as specified by model.",
)
@click.option(
    '--host', 'host_url', type=str,
    default=DEFAULT_HOST_URL, envvar='HOST_URL',
    help="Host base url."
)
@click.option(
    '--port', 'host_port', type=int,
    default=DEFAULT_HOST_PORT, envvar='HOST_PORT',
    help="Host base port."
)
@click.option(
    '--model_dir', type=str,
    default=DEFAULT_MODEL_DIR, envvar='MODEL_DIR',
    help="Directory containing models."
)
@click.option(
    '--model_name', type=str,
    default=DEFAULT_MODEL_NAME, envvar='MODEL_NAME',
    help="Model name for reference (must be in model directory).",
)
@click.option(
    '--url', 'endpoint', type=str,
    help="""Full url to the prediction endpoint.
    If not provided, default constitutes from model_dir and model_name settings.""",
)
@click.option(
    '--indent', type=int,
    default=0,
    help="Whether to indent output."
)
def predict(input_file: str,
            signature: str = None,
            host_url: str = None,
            host_port: int = None,
            model_dir: str = None,
            model_name: str = None,
            endpoint: str = None,
            indent: int = None):
    """Query served model for predictions.

    INPUT_FILE: Path to file containing JSON instances to be fed to the model.
    """
    # send request
    instances: list = json.loads(Path(input_file).read_text())

    for (arg, exp_type) in {'signature': str, 'instances': list}.items():
        instance = eval(arg)
        if not isinstance(instance, exp_type):
            raise TypeError("`{arg}` expected to be {expected}, got: {got}".format(
                arg=arg, expected=exp_type, got=type(instance)
            ))

    data = format_data(instances=instances, signature=signature)
    headers = {'content-type': 'application/json'}

    # slash commonly passed to directories, strip the end-slash from directory path
    model_dir = model_dir[:-1] if model_dir.endswith('/') else model_dir

    endpoint = endpoint or f'{host_url}:{host_port}/v1/{model_dir}/{model_name}:predict'
    response = requests.post(endpoint, data=data, headers=headers)
    print(response)

    # echo predictions
    if indent:
        click.echo(json.dumps(json.loads(response.text), indent=indent))
    else:
        click.echo(response.json())

    sys.exit(0)


def format_data(instances: List[List[float]], signature: str = "serving_default") -> str:
    """Format data according to JSON data model."""
    data_model = _DEFAULT_DATA_MODEL
    data_model.update(signature=signature, instances=instances)

    return json.dumps(data_model)


if __name__ == '__main__':
    predict()

