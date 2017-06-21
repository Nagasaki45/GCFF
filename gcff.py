"""
A python wrapper around the GCFF algorithm (originally in Matlab, see gc.m).

The API contains the `gc` function, for single frame analysis, and a
`ContinuousGC` class for tracking F-Formations over time.

For more information refer to the paper
http://vips.sci.univr.it/research/fformation/.
"""

import collections
import itertools
from time import monotonic

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


class ContinuousGC:
    """
    Track F-Formations in time. Each F-Formation is given an
    ID and a set of members. New F-Formation are created when
    new one is found outside of a "max change" radius of any
    previous one.

    Usage:

        gc = ContinuousGC(stride=1, mdl=10, max_change_per_second=1)
        gc.update(my_features)
        # A list of dicts with 'id', 'members', and 'centre' keys.
        print(gc.f_formations)
    """

    def __init__(self, stride, mdl, max_change_per_second):
        """
        For `stride` and `mdl` information see the `gc` function and the
        original publication.

        max_change_per_second: the maximum distance an F-Formation can
                               move in 1 second and still be considered
                               as the same conversational group (with same
                               ID).
        """
        self.stride = stride
        self.mdl = mdl
        self.max_change_per_second = max_change_per_second

        self.f_formations = []
        self._previous_update_time = None
        self._id_generator = itertools.count()


    def update(self, features, time=None):
        """
        features: see the `gc` function for more information.
        time: the time in fraction of seconds. If ommited monotonic time is
              used.
        """
        if time is None:
            time = monotonic()
        if self._previous_update_time is None:
            max_change = float('inf')
        else:
            time_diff = time - self._previous_update_time
            max_change = self.max_change_per_second * time_diff

        seg = gc(features, self.stride, self.mdl)
        keys = get_keys(features)
        new_ffs = [{'members': members, 'centre': get_centre(features, members)}
                    for members in group_results(seg, keys)]

        for new_ff in new_ffs:
            for old_ff in self.f_formations:
                if dist(old_ff['centre'], new_ff['centre']) < max_change:
                    new_ff['id'] = old_ff['id']
                    break
            else:
                new_ff['id'] = next(self._id_generator)

        self.f_formations = new_ffs
        self._previous_update_time = time


#####################
# Utility functions #
#####################


def get_keys(features):
    """
    Get the members IDs from the features matrix.
    """
    return features[:, 0].astype(int)


def group_results(seg, keys):
    """
    Convert the inconvenient list of F-Formation assignment to list of lists
    of IDs, each one represent an F-Formations.
    """
    d = collections.defaultdict(list)
    for assignment, key in zip(seg, keys):
        d[assignment].append(key)
    return list(d.values())


def get_centre(features, members):
    """
    Find the x, y centre of a given list of members IDs.
    """
    position_columns = slice(1, 3)
    mask = np.in1d(get_keys(features), members)
    positions = features[mask, position_columns]
    return positions.mean(axis=0)


def dist(x, y):
    return np.square(np.sum((x - y) ** 2))
