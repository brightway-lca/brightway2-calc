# -*- coding: utf-8 -*-
from bw2calc.matrices import (
    build_labelled_matrix,
    flip_amounts,
    build_matrix,
    build_diagonal_matrix,
)
from bw_processing import MAX_SIGNED_32BIT_INT as M
import numpy as np
import pytest


def test_flip_amounts():
    dtype = [
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, False), (2, True)], dtype=dtype)
    expected = [1, -2]
    result = flip_amounts(array)
    assert np.allclose(result, expected)


def test_flip_amounts_in_place():
    dtype = [
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, False), (2, True)], dtype=dtype)
    flip_amounts(array)
    assert np.allclose(array["amount"], [1, -2])


def test_flip_amounts_in_place_separate_vector():
    dtype = [
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, False), (2, True)], dtype=dtype)
    result = flip_amounts(array, np.array([1, 2]))
    assert np.allclose(array["amount"], [1, 2])
    assert np.allclose(result, [1, -2])


def test_flip_amounts_missing_field():
    dtype = [
        ("amount", np.float32),
    ]
    array = np.array([(1,), (2,)], dtype=dtype)
    with pytest.raises(ValueError):
        flip_amounts(array, np.array([1, 2]))


def test_build_diagonal_matrix():
    dtype = [
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 99, False), (1, 100, True)], dtype=dtype)
    matrix = build_diagonal_matrix(array, 2)
    assert matrix.shape == (2, 2)
    assert matrix.sum() == -1


def test_build_diagonal_matrix_ignores_column_indices():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, 0, 99, False), (0, 1, 100, False),], dtype=dtype)
    matrix = build_diagonal_matrix(array, 2)
    assert np.allclose(matrix.toarray(), np.array(((100, 0), (0, 99))))


def test_build_diagonal_matrix_new_data():
    dtype = [
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 99, False), (1, 100, True)], dtype=dtype)
    matrix = build_diagonal_matrix(array, 2, np.array([111, 101]))
    assert matrix.shape == (2, 2)
    assert matrix.sum() == 10


def test_build_diagonal_matrix_one_by_one():
    dtype = [
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 99, False)], dtype=dtype)
    matrix = build_diagonal_matrix(array, 1)
    print(matrix.toarray())
    assert matrix.shape == (1, 1)
    assert matrix.sum() == 99


def test_build_diagonal_matrix_zero_by_zero():
    dtype = [
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([], dtype=dtype)
    matrix = build_diagonal_matrix(array, 0)
    print(matrix.toarray())
    assert matrix.shape == (0, 0)
    assert not matrix.sum()


def test_build_matrix():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    matrix = build_matrix(array, 2, 3)
    expected = np.array([(-1, 0, 0), (10, -2, 11)])
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix.toarray(), expected)


def test_build_matrix_one_by_one():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 0, 10, False)], dtype=dtype)
    matrix = build_matrix(array, 1, 1)
    assert matrix.shape == (1, 1)
    assert matrix.sum() == 10


def test_build_matrix_zero_by_zero():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([], dtype=dtype)
    matrix = build_matrix(array, 0, 0)
    assert matrix is not None
    assert matrix.shape == (0, 0)
    assert not matrix.sum()


def test_build_matrix_empty_rows_cols():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    matrix = build_matrix(array, 10, 11)
    assert matrix.shape == (10, 11)
    assert matrix.sum() == 18


def test_build_matrix_custom_labels():
    dtype = [
        ("ri", np.int32),
        ("ci", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    matrix = build_matrix(array, 2, 3, "ri", "ci")
    assert matrix.shape == (2, 3)
    assert matrix.sum() == 18


def test_build_matrix_no_flip():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    matrix = build_matrix(array, 2, 3, flip=False)
    assert matrix.shape == (2, 3)
    assert matrix.sum() == 24


def test_build_matrix_new_data():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    new_data = np.array((1, 2, 3, 4))
    matrix = build_matrix(array, 2, 3, new_data=new_data)
    assert matrix.shape == (2, 3)
    assert matrix.sum() == 4


def test_build_matrix_new_data_no_flip():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1, True), (1, 1, 2, True), (1, 0, 10, False), (1, 2, 11, False),],
        dtype=dtype,
    )
    new_data = np.array((1, 2, 3, 4))
    matrix = build_matrix(array, 2, 3, new_data=new_data, flip=False)
    assert matrix.shape == (2, 3)
    assert matrix.sum() == 10


def test_build_matrix_sum_multiple_values():
    dtype = [
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [
            (0, 0, 1, True),
            (1, 1, 2, True),
            (1, 0, 10, False),
            (1, 0, 100, False),
            (1, 2, 11, False),
        ],
        dtype=dtype,
    )
    matrix = build_matrix(array, 2, 3)
    expected = np.array([(-1, 0, 0), (110, -2, 11)])
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix.toarray(), expected)


def test_build_labelled_matrices():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 0, M, M, 1, True), (1, 1, M, M, 11, False),], dtype=dtype)
    _, _, _, matrix = build_labelled_matrix(array)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == -1
    assert matrix[1, 1] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_zero_dimensions():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([], dtype=dtype)
    _, _, _, matrix = build_labelled_matrix(array)
    assert matrix.shape == (0, 0)


def test_build_labelled_matrices_offset_input_indices():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(1000, 100, M, M, 1, True), (1001, 101, M, M, 11, False),], dtype=dtype,
    )
    _, _, _, matrix = build_labelled_matrix(array)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == -1
    assert matrix[1, 1] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_offset_noncontiguous_indices():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(1000, 100, M, M, 1, True), (2000, 200, M, M, 11, False),], dtype=dtype,
    )
    _, _, _, matrix = build_labelled_matrix(array)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == -1
    assert matrix[1, 1] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_drop_missing():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 1, M, M, 1, True), (1, 0, M, M, 11, False), (2, 0, M, M, 11, False),],
        dtype=dtype,
    )
    row_dict = {0: 0, 1: 1}
    _, _, _, matrix = build_labelled_matrix(array, row_dict=row_dict)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == -1
    assert matrix[1, 0] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_no_drop_missing_error():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 1, M, M, 1, True), (1, 0, M, M, 11, False), (2, 0, M, M, 11, False),],
        dtype=dtype,
    )
    row_dict = {0: 0, 1: 1}
    with pytest.raises(ValueError):
        build_labelled_matrix(array, row_dict=row_dict, drop_missing=False)


def test_build_labelled_matrices_only_row_dict():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(10, 1, M, M, 1, True), (11, 0, M, M, 11, False),], dtype=dtype)
    row_dict = {10: 1, 11: 0}
    _, _, _, matrix = build_labelled_matrix(array, row_dict=row_dict)
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] == -1
    assert matrix[0, 0] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_only_col_dict():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 11, M, M, 1, True), (1, 10, M, M, 11, False),], dtype=dtype)
    col_dict = {10: 2, 11: 1}
    _, _, _, matrix = build_labelled_matrix(array, col_dict=col_dict)
    assert matrix.shape == (2, 3)
    assert matrix[0, 1] == -1
    assert matrix[1, 2] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_offset_dict():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(0, 0, M, M, 1, True), (1, 2, M, M, 11, False),], dtype=dtype)
    row_dict = {0: 10, 1: 11}
    col_dict = {0: 20, 1: 21, 2: 22}
    _, _, _, matrix = build_labelled_matrix(array, row_dict, col_dict)
    assert matrix.shape == (12, 23)
    assert matrix[10, 20] == -1
    assert matrix[11, 22] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_mapping_ignores_current_indices():
    dtype = [
        ("row_value", np.int32),
        ("col_value", np.int32),
        ("row_index", np.int32),
        ("col_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(0, 0, 1000, 1000, 1, True), (1, 2, 1000, 1000, 11, False),], dtype=dtype,
    )
    row_dict = {0: 0, 1: 1}
    col_dict = {0: 0, 1: 1, 2: 2}
    _, _, _, matrix = build_labelled_matrix(array, row_dict, col_dict)
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == -1
    assert matrix[1, 2] == 11
    assert matrix.sum() == 10


def test_build_labelled_matrices_offset_dict_one_d():
    dtype = [
        ("row_value", np.int32),
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, M, 99, False), (2, M, 100, False),], dtype=dtype)
    row_dict = {1: 10, 2: 11}
    array, _, _, matrix = build_labelled_matrix(array, row_dict=row_dict, one_d=True,)
    assert np.allclose(array["row_value"], [1, 2])
    assert np.allclose(array["row_index"], [10, 11])
    assert matrix.shape == (12, 12)
    assert np.allclose(matrix.toarray()[10:, 10:], np.array(((99, 0), (0, 100))))
    assert matrix.sum() == 199


def test_build_labelled_matrices_one_d():
    dtype = [
        ("row_value", np.int32),
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array([(1, M, 99, False), (2, M, 100, False),], dtype=dtype)
    row_dict = {1: 0, 2: 1}
    array, _, _, matrix = build_labelled_matrix(array, row_dict=row_dict, one_d=True,)
    assert np.allclose(array["row_value"], [1, 2])
    assert np.allclose(array["row_index"], [0, 1])
    assert np.allclose(matrix.toarray(), np.array(((99, 0), (0, 100))))


def test_build_labelled_matrices_one_d_drop_missing():
    dtype = [
        ("row_value", np.int32),
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(1, M, 99, False), (2, M, 100, False), (3, M, 101, False),], dtype=dtype,
    )
    row_dict = {1: 0, 2: 1}
    array, _, _, matrix = build_labelled_matrix(array, row_dict=row_dict, one_d=True,)
    assert np.allclose(array["row_value"], [1, 2])
    assert np.allclose(array["row_index"], [0, 1])
    assert np.allclose(matrix.toarray(), np.array(((99, 0), (0, 100))))


def test_build_labelled_matrices_one_d_no_drop_missing_right_dimensions():
    dtype = [
        ("row_value", np.int32),
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(1, M, 99, False), (2, M, 100, False), (3, M, 101, False),], dtype=dtype,
    )
    row_dict = {1: 0, 2: 1, 10: 3}
    with pytest.raises(ValueError):
        build_labelled_matrix(array, row_dict=row_dict, one_d=True, drop_missing=False)


def test_build_labelled_matrices_one_d_no_drop_missing_wrong_dimensions():
    dtype = [
        ("row_value", np.int32),
        ("row_index", np.int32),
        ("amount", np.float32),
        ("flip", np.bool),
    ]
    array = np.array(
        [(1, M, 99, False), (2, M, 100, False), (3, M, 101, False),], dtype=dtype,
    )
    row_dict = {1: 0, 2: 1}
    with pytest.raises(ValueError):
        build_labelled_matrix(array, row_dict=row_dict, one_d=True, drop_missing=False)
