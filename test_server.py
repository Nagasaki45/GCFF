import unittest

import numpy as np

import server


class TestServer(unittest.TestCase):

    def test_request_parsing(self):
        features = server.parse_features("1,2,3,4\n5,6,7,8".encode())
        expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        np.testing.assert_equal(features, expected)

    def test_request_parsing_with_one_participant(self):
        features = server.parse_features("1,2,3,4".encode())
        expected = np.array([[1, 2, 3, 4]])
        np.testing.assert_equal(features, expected)
