"""
An http endpoint for the GCFF algorithm.
"""

import io

from flask import Flask, request
import numpy as np

import gcff
from server_settings import STRIDE, MDL


app = Flask(__name__)


@app.route('/', methods=['POST'])
def gcff_handler():
    binary_io = io.BytesIO(request.data)
    features = np.genfromtxt(binary_io, delimiter=',')
    seg = gcff.gc(features, stride=STRIDE, mdl=MDL)
    return ','.join(str(x) for x in seg)


if __name__ == "__main__":
    app.run()
