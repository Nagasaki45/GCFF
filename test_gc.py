import unittest

import numpy as np

import gcff


class TestUtilities(unittest.TestCase):

    def test_get_keys(self):
        features = np.array([
            [1, 7, 8, 9],
            [2, 7, 8, 9],
        ])
        np.testing.assert_equal(gcff.get_keys(features), [1, 2])

    def test_group_results(self):
        keys = [1, 3, 5]
        seg = [0, 0, 1]
        expected = [[1, 3], [5]]
        self.assertEqual(gcff.group_results(seg, keys), expected)

    def test_get_centre(self):
        features = np.array([
            [1, 1, 2, 0],
            [2, 3, 4, np.pi],
        ])
        result = gcff.get_centre(features, [1, 2])
        np.testing.assert_equal(result, np.array([2, 3]))


class TestContinuousGC(unittest.TestCase):

    def setUp(self):
        self.gc = gcff.ContinuousGC(stride=1, mdl=10, max_change_per_second=1)

    def test_single_update(self):
        # Two members facing each other
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
        ])
        self.gc.update(features, time=0)
        self.assertEqual(len(self.gc.f_formations), 1)
        self.assertEqual(self.gc.f_formations[0]['members'], [1, 2])

    def test_joining_an_f_formation(self):
        # Two members facing each other
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
        ])
        self.gc.update(features, time=0)
        ff_id = self.gc.f_formations[0]['id']

        # Another member joins
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
            [3, 1, 1.5, 1.5 * np.pi],
        ])
        self.gc.update(features, time=1)

        self.assertEqual(len(self.gc.f_formations), 1)
        self.assertEqual(self.gc.f_formations[0]['id'], ff_id)

    def test_leaving_an_f_formation(self):
        # 3 members conversation
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
            [3, 1, 1.5, 1.5 * np.pi],
        ])
        self.gc.update(features, time=0)
        ff_id = self.gc.f_formations[0]['id']

        # Member 3 leaves
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
            [3, 3, 0, 0],
        ])
        self.gc.update(features, time=1)

        self.assertEqual(len(self.gc.f_formations), 2)
        for ff in self.gc.f_formations:
            if ff['id'] == ff_id:
                self.assertEqual(ff['members'], [1, 2])
                break
        else:
            assert False, 'This should never happen'

    def test_breaking_an_f_formation(self):
        # 2 members conversation
        features = np.array([
            [1, 0, 0, 0],
            [2, 2, 0, np.pi],
        ])
        self.gc.update(features, time=0)
        ff_id = self.gc.f_formations[0]['id']

        # Both turn around
        features = np.array([
            [1, 0, 0, np.pi],
            [2, 2, 0, 0],
        ])
        self.gc.update(features, time=1)

        self.assertEqual(len(self.gc.f_formations), 2)
        for ff in self.gc.f_formations:
            self.assertNotEqual(ff['id'], ff_id)


if __name__ == '__main__':
    unittest.main()
