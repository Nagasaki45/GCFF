"""
An http endpoint for the GCFF algorithm. The stride and mdl
should be given as command line arguments.

Usage:

  python server.py 1 100    # stride = 1, mdl = 100
  python server.py 1 100 3  # stride = 1, mdl = 100, max_change_per_second = 3
"""

import json
import io
import sys

from flask import Flask, request
import numpy as np

import gcff

app = Flask(__name__)


@app.route('/', methods=['POST'])
def gcff_handler():
    features = parse_features(request.data)
    seg = gcff.gc(features, stride=stride, mdl=mdl)
    return ','.join(str(x) for x in seg)


@app.route('/continuous', methods=['POST'])
def continuous_handler():
    features = parse_features(request.data)
    continuous.update(features)
    return json.dumps(continuous.f_formations, cls=NumpyJSONEncoder)


class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_features(data):
    binary_io = io.BytesIO(data)
    features = np.genfromtxt(binary_io, delimiter=',')
    if features.ndim < 2:
        return features[np.newaxis, :]
    return features


if __name__ == "__main__":
    stride = float(sys.argv[1])
    mdl = float(sys.argv[2])
    try:
        max_change_per_second = float(sys.argv[3])
    except IndexError:
        max_change_per_second = 1

    continuous = gcff.ContinuousGC(stride, mdl, max_change_per_second)

    app.run()
