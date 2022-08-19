try:
    import bw2data as bd
    import bw2io as bi
    from bw2data.tests import bw2test
except ImportError:
    bd = None
import bw2calc as bc
import pytest
from pathlib import Path
import json
from pandas.testing import assert_frame_equal
import pandas as pd

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def frames(one, two):
    assert_frame_equal(
        one.reindex(sorted(one.columns), axis=1),
        two.reindex(sorted(two.columns), axis=1),
        rtol=1e-04,
        atol=1e-04,
        check_dtype=False,
    )


@pytest.fixture
def basic_example():
    mapping = dict(json.load(open(fixture_dir / "bw2io_example_db_mapping.json")))
    packages = [
        fixture_dir / "bw2io_example_db.zip",
        fixture_dir / "ipcc_simple.zip",
    ]

    lca = bc.LCA(
        {mapping["Driving an electric car"]: 1},
        data_objs=packages,
    )
    lca.lci()
    lca.lcia()

    return lca, mapping


def test_to_dataframe_basic(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    steel = mapping['Steel']
    co2 = mapping['CO2']

    df = lca.to_dataframe(annotate=False)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.16800001296144934,
        'col_id': elec,
        'col_index': lca.dicts.activity[elec],
    }, {
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.014481599635022317,
        'col_id': steel,
        'col_index': lca.dicts.activity[steel],
    }])
    frames(expected, df)


def test_to_dataframe_inventory_matrix(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    steel = mapping['Steel']
    co2 = mapping['CO2']

    df = lca.to_dataframe(matrix_label='inventory', annotate=False)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.16800001296144934,
        'col_id': elec,
        'col_index': lca.dicts.activity[elec],
    }, {
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.014481599635022317,
        'col_id': steel,
        'col_index': lca.dicts.activity[steel],
    }])
    frames(expected, df)


def test_to_dataframe_characterization_matrix(basic_example):
    lca, mapping = basic_example

    co2 = mapping['CO2']

    df = lca.to_dataframe(matrix_label='characterization_matrix', annotate=False)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 1,
        'col_id': co2,
        'col_index': lca.dicts.biosphere[co2],
    }])
    frames(expected, df)


def test_to_dataframe_technosphere_matrix(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    steel = mapping['Steel']
    batt = mapping['Electric car battery']
    ecar = mapping['Electric car']
    ccar = mapping['Combustion car']

    df = lca.to_dataframe(matrix_label='technosphere_matrix', annotate=False, cutoff=4)

    expected = pd.DataFrame([{
        'row_id': elec,
        'row_index': lca.dicts.product[elec],
        'amount': -2000,
        'col_id': batt,
        'col_index': lca.dicts.activity[batt],
    }, {
        'row_id': steel,
        'row_index': lca.dicts.product[steel],
        'amount': -1921,
        'col_id': ecar,
        'col_index': lca.dicts.activity[ecar],
    }, {
        'row_id': steel,
        'row_index': lca.dicts.product[steel],
        'amount': -1641,
        'col_id': ccar,
        'col_index': lca.dicts.activity[ccar],
    }, {
        'row_id': steel,
        'row_index': lca.dicts.product[steel],
        'amount': -9.880000114440918,
        'col_id': batt,
        'col_index': lca.dicts.activity[batt],
    }])
    frames(expected, df)


def test_to_dataframe_biosphere_matrix(basic_example):
    lca, mapping = basic_example

    co2 = mapping['CO2']
    elec = mapping['Electricity']
    steel = mapping['Steel']
    ccar = mapping['Driving an combustion car']

    df = lca.to_dataframe(matrix_label='biosphere_matrix', annotate=False)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 1.5,
        'col_id': steel,
        'col_index': lca.dicts.activity[steel],
    }, {
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.6,
        'col_id': elec,
        'col_index': lca.dicts.activity[elec],
    }, {
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.1426,
        'col_id': ccar,
        'col_index': lca.dicts.activity[ccar],
    }])
    frames(expected, df)


def test_to_dataframe_number_cutoff(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    co2 = mapping['CO2']

    df = lca.to_dataframe(annotate=False, cutoff=1)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.16800001296144934,
        'col_id': elec,
        'col_index': lca.dicts.activity[elec],
    }])
    frames(expected, df)


def test_to_dataframe_fraction_cutoff(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    steel = mapping['Steel']
    co2 = mapping['CO2']

    df = lca.to_dataframe(annotate=False)

    expected = pd.DataFrame([{
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.16800001296144934,
        'col_id': elec,
        'col_index': lca.dicts.activity[elec],
    }, {
        'row_id': co2,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.014481599635022317,
        'col_id': steel,
        'col_index': lca.dicts.activity[steel],
    }])
    frames(expected, df)


def test_to_dataframe_custom_mappings(basic_example):
    lca, mapping = basic_example

    elec = mapping['Electricity']
    steel = mapping['Steel']
    co2 = mapping['CO2']

    df = lca.to_dataframe(
        annotate=False,
        row_dict={lca.dicts.biosphere[co2]: 111},
        col_dict={
            lca.dicts.activity[steel]: 201,
            lca.dicts.activity[elec]: 202,
        }
    )

    expected = pd.DataFrame([{
        'row_id': 111,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.16800001296144934,
        'col_id': 202,
        'col_index': lca.dicts.activity[elec],
    }, {
        'row_id': 111,
        'row_index': lca.dicts.biosphere[co2],
        'amount': 0.014481599635022317,
        'col_id': 201,
        'col_index': lca.dicts.activity[steel],
    }])
    print(df)
    frames(expected, df)


@pytest.mark.skipif(not bd, reason="bw2data not installed")
def test_foo():
    pass


# bi.add_example_database()
