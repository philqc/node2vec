import unittest
from src.preprocess import *
from src.utils import project_root
import os

TEST_CSV = os.path.join(project_root(), "tests", "data", "Relation", "Relation.csv")


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = load_csv(TEST_CSV)

    def test_neighbors(self):
        probs = get_transition_probabilites(self.df, False)

        for previous, possible_starts in probs.items():
            # Make sure we have a probability distribution accross neighbors
            for start, neighbors in possible_starts.items():
                self.assertAlmostEqual(sum(neighbors.values()), 1)
                # make sure we can always go back to previous node
                self.assertTrue(previous in neighbors.keys())

        # Use our prior knowledge abour Relation.csv to make sure this make sense
        neighbors = probs["user1"]["page1"]
        self.assertTrue("user2" in neighbors.keys())
        self.assertTrue("user6" in neighbors.keys())
        self.assertTrue(len(neighbors) == 3)

        neighbors = probs["user2"]["page4"]
        self.assertTrue(len(neighbors) == 1)
