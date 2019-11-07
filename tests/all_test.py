import unittest
from src.preprocess import *
from src.learn_features import random_walk
from src.utils import project_root
import os

TEST_CSV = os.path.join(project_root(), "tests", "data", "Relation", "Relation.csv")


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = load_csv(TEST_CSV)
        self.matrix_probs, self.all_nodes = get_transition_probabilites(self.df, False, min_like=2,
                                                                        drop_page_ids=False)

    def test_neighbors(self):

        for previous, possible_starts in self.matrix_probs.items():
            for start, neighbors in possible_starts.items():
                # Make sure we have a probability distribution accross neighbors
                self.assertAlmostEqual(sum(neighbors.values()), 1)
                # make sure we can always go back to previous node
                self.assertTrue(previous in neighbors.keys())

        # Use our prior knowledge abour Relation.csv to make sure this make sense
        neighbors = self.matrix_probs["user1"]["page1"]
        self.assertTrue("user2" in neighbors.keys())
        self.assertTrue("user6" in neighbors.keys())
        self.assertTrue(len(neighbors) == 3)

        neighbors = self.matrix_probs["user2"]["page4"]
        self.assertTrue(len(neighbors) == 1)

    def test_random_walk(self):
        """ Make sure our monte carlo estimates of probability distribution
        from random sampling is equal to the true probability distribution
        """
        mc_estimate = {
            "page4": 0,
            "page1": 0,
            "page2": 0
        }

        length = 1
        walk = random_walk(self.matrix_probs, "page4", length)
        self.assertEqual(len(walk), length + 1)

        for i in range(10000):
            walk = random_walk(self.matrix_probs, "page4", length)
            mc_estimate[walk[-1]] += 1

        mc_estimate = prob_distribution_from_dict(mc_estimate)
        real_prob_distribution = {
            "page4": 1 / PARAMETERS["p"],
            "page1": 1 / PARAMETERS["q"],
            "page2": 1 / PARAMETERS["q"]
        }
        real_prob_distribution = prob_distribution_from_dict(real_prob_distribution)
        for key in mc_estimate.keys():
            self.assertAlmostEqual(mc_estimate[key], real_prob_distribution[key], places=1)

    def test_node_list(self):
        self.assertEqual(len(list_all_nodes(self.df)), 10)
