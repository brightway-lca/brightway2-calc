from ..errors import OutsideTechnosphere
from ..lca import LCA
from bw2data import *
from bw2data.tests import BW2DataTest
import numpy as np


class LCACalculationTestCase(BW2DataTest):
    def add_basic_biosphere(self):
        biosphere = Database("biosphere")
        biosphere.register()
        biosphere.write({
            ("biosphere", "1"): {
                'categories': ['things'],
                'exchanges': [],
                'name': 'an emission',
                'type': 'emission',
                'unit': 'kg'
                }})
        biosphere.process()

    def test_basic(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 0.5,
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        answer = np.zeros((2,))
        answer[lca.technosphere_dict[mapping[("t", "1")]]] = 1
        answer[lca.technosphere_dict[mapping[("t", "2")]]] = 0.5
        self.assertTrue(np.allclose(answer, lca.supply_array))

    def test_redo_lci_fails_if_activity_outside_technosphere(self):
        self.add_basic_biosphere()
        test_data = {("t", "1"): {
            'exchanges': [
                {'amount': 1, 'input': ('biosphere', "1"), 'type': 'biosphere'}
        ]}}
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        more_test_data = {("z", "1"): {
            'exchanges': [
                {'amount': 1, 'input': ('t', "1"), 'type': 'technosphere'}
        ]}}
        more_test_db = Database("z")
        more_test_db.register()
        more_test_db.write(more_test_data)
        more_test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        with self.assertRaises(OutsideTechnosphere):
            lca.redo_lci({("z", "1"): 1})

    def test_production_values(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 2,
                    'input': ('t', "1"),
                    'type': 'production',
                    'uncertainty type': 0},
                    {'amount': 0.5,
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        answer = np.zeros((2,))
        answer[lca.technosphere_dict[mapping[("t", "1")]]] = 0.5
        answer[lca.technosphere_dict[mapping[("t", "2")]]] = 0.25
        self.assertTrue(np.allclose(answer, lca.supply_array))

    def test_substitution(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 1,
                    'input': ('t', "2"),
                    'type': 'substitution',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        answer = np.zeros((2,))
        answer[lca.technosphere_dict[mapping[("t", "1")]]] = 1
        answer[lca.technosphere_dict[mapping[("t", "2")]]] = -1
        self.assertTrue(np.allclose(answer, lca.supply_array))

    def test_substitution_no_type(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': -1,  # substitution
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        answer = np.zeros((2,))
        answer[lca.technosphere_dict[mapping[("t", "1")]]] = 1
        answer[lca.technosphere_dict[mapping[("t", "2")]]] = -1
        self.assertTrue(np.allclose(answer, lca.supply_array))

    def test_circular_chains(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 0.5,
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [{
                    'amount': 0.1,
                    'input': ('t', "1"),
                    'type': 'technosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()
        answer = np.zeros((2,))
        answer[lca.technosphere_dict[mapping[("t", "1")]]] = 20 / 19.
        answer[lca.technosphere_dict[mapping[("t", "2")]]] = 10 / 19.
        self.assertTrue(np.allclose(answer, lca.supply_array))

    def test_dependent_databases(self):
        databases['one'] = {'depends': ['two', 'three']}
        databases['two'] = {'depends': ['four', 'five']}
        databases['three'] = {'depends': ['four']}
        databases['four'] = {'depends': ['six']}
        databases['five'] = {'depends': ['two']}
        databases['six'] = {'depends': []}
        lca = LCA({('one', None): 1})
        self.assertEqual(
            lca.databases,
            {'one', 'two', 'three', 'four', 'five', 'six'}
        )

    def test_demand_type(self):
        with self.assertRaises(ValueError):
            LCA(("foo", "1"))
        with self.assertRaises(ValueError):
            LCA("foo")
        with self.assertRaises(ValueError):
            LCA([{"foo": "1"}])

    def test_decomposed_uses_solver(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 0.5,
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci(factorize=True)
        # Indirect test because no easy way to test a function is called
        lca.technosphere_matrix = None
        self.assertEqual(float(lca.solve_linear_system().sum()), 1.5)

    def test_fix_dictionaries(self):
        test_data = {
            ("t", "1"): {
                'exchanges': [{
                    'amount': 0.5,
                    'input': ('t', "2"),
                    'type': 'technosphere',
                    'uncertainty type': 0},
                    {'amount': 1,
                    'input': ('biosphere', "1"),
                    'type': 'biosphere',
                    'uncertainty type': 0}],
                'type': 'process',
                'unit': 'kg'
                },
            ("t", "2"): {
                'exchanges': [],
                'type': 'process',
                'unit': 'kg'
                },
            }
        self.add_basic_biosphere()
        test_db = Database("t")
        test_db.register()
        test_db.write(test_data)
        test_db.process()
        lca = LCA({("t", "1"): 1})
        lca.lci()

        supply = lca.supply_array.sum()

        self.assertTrue(lca._mapped_dict)
        self.assertTrue(lca.fix_dictionaries())
        self.assertFalse(lca._mapped_dict)
        # Second time doesn't do anything
        self.assertFalse(lca.fix_dictionaries())
        self.assertFalse(lca._mapped_dict)
        lca.redo_lci({("t", "1"): 2})
        self.assertEqual(lca.supply_array.sum(), supply * 2)
