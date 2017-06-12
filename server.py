"""
An http endpoint for the GCFF algorithm. The stride and mdl
should be given as command line arguments.

Usage:

  python server.py 1 100  # stride = 1, mdl = 100
"""

import io
import sys

from flask import Flask, request
import numpy as np

import gcff

stride = float(sys.argv[1])
mdl = float(sys.argv[2])


app = Flask(__name__)


@app.route('/', methods=['POST'])
def gcff_handler():
    binary_io = io.BytesIO(request.data)
    features = np.genfromtxt(binary_io, delimiter=',')
    seg = gcff.gc(features, stride=stride, mdl=mdl)
    return ','.join(str(x) for x in seg)


if __name__ == "__main__":
    app.run()
