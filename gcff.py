"""
A python wrapper around the GCFF algorithm (originally in Matlab, see gc.m).

For more information refer to the paper
http://vips.sci.univr.it/research/fformation/.
"""

from oct2py import octave
import numpy as np

def gc(features , stride, mdl):
    """
    F-Formation analysis using the GCFF algorithm.

    features: a N*4 numpy.ndarray with rows representing
              [id, x_position, y_position, orientation].
              orientation is given in radians.
    stride: the distance between the individual i and the centre
            of its transactional segment.
    mdl: a minimum description length prior, linearly penalising the
         log-likelihood for the number of models used.

    Returns a vector of length N with the o-space assignment of each
    participant.
    """

    stride = float(stride)
    mdl = float(mdl)
    result = octave.gc(features, stride, mdl)
    # In case of one participant octave returns a single float
    if isinstance(result, float):
        return np.array([result], dtype=int)
    # Octave returns a n*1 column (2 dimensional in numpy terms)
    result = result.T[0]
    # It came back as np.float64, should be ints
    result = result.astype(int)
    return result
