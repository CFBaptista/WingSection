import numpy as np
import pytest

from wingsection.airfoil import Airfoil, close_polygon, open_polygon

_CONSTRUCTION_NAME = 'test'
_FORWARD_ROTATION_ANGLE = 0.63
_FORWARD_SCALE = 2.5
_FORWARD_TRANSLATION = np.array([3.0, -4.0])
_ALIGNMENT_TOLERANCE = 1e-14
_COORDINATE_SCALE = 2.0
_UPDATED_TRAILING_Y = 2.0
_ORIGIN = np.array([0.0, 0.0])
_UNIT_TRAILING_POINT = np.array([1.0, 0.0])
_TWO_DIMENSIONAL_IDENTITY = np.eye(2)

_VALID_COORDINATES = np.array(
    [
        [1.0, 0.6],
        [0.25, 0.7],
        [0.0, 0.5],
        [0.25, 0.3],
        [1.0, 0.4],
        [1.0, 0.5],
    ],
)

_EXPECTED_TANGENTS = np.array(
    [
        [-0.9912279006826347, 0.13216372009101793],
        [-0.9950371902099893, -0.09950371902099892],
        [0.0, -1.0],
        [0.9950371902099893, -0.09950371902099892],
        [0.9662349396012462, 0.25766265056033233],
        [0.0, 1.0],
    ],
)

_EXPECTED_NORMALS = np.array(
    [
        [0.13216372009101793, 0.9912279006826347],
        [-0.09950371902099892, 0.9950371902099893],
        [-1.0, 0.0],
        [-0.09950371902099892, -0.9950371902099893],
        [0.25766265056033233, -0.9662349396012462],
        [1.0, 0.0],
    ],
)

_EXPECTED_LENGTHS = np.array(
    [
        0.7566372975210778,
        0.3201562118716424,
        0.32015621187164245,
        0.7566372975210778,
        0.09999999999999998,
    ],
)

_ONE_DIMENSIONAL_INVALID_COORDINATES = _VALID_COORDINATES.flatten()
_WRONG_COLUMN_COUNT_COORDINATES = _VALID_COORDINATES[:, :1]
_TOO_FEW_COORDINATES = _VALID_COORDINATES[:-2]
_INTEGER_COORDINATES = _VALID_COORDINATES.astype(int)
_COMPLEX_COORDINATES = _VALID_COORDINATES.astype(complex)
_STRING_COORDINATES = _VALID_COORDINATES.astype(str)
_NON_FINITE_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [np.nan, 1.0],
        [0.0, 1.0],
    ],
)
_INFINITE_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, np.inf],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
)
_DUPLICATE_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ],
)
_CLOCKWISE_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.0],
    ],
)
_COLLINEAR_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [0.25, 0.0],
        [0.5, 0.0],
        [0.75, 0.0],
        [1.0, 0.0],
    ],
)


def _valid_coordinates() -> np.ndarray:
    return _VALID_COORDINATES.copy()


def test_airfoil_construction() -> None:
    # GIVEN: valid airfoil coordinates, a name, and a copy of the expected coordinates.

    coordinates = _valid_coordinates()
    name = _CONSTRUCTION_NAME
    expected_coordinates = coordinates.copy()

    # WHEN: an airfoil is constructed.

    airfoil = Airfoil(coordinates, name)
    actual_coordinates = airfoil.coordinates
    actual_dtype = actual_coordinates.dtype
    actual_name = airfoil.name

    # THEN: the actual coordinates, dtype, and name match the expected values.

    assert actual_coordinates is not coordinates
    np.testing.assert_array_equal(actual_coordinates, expected_coordinates)
    assert actual_dtype == coordinates.dtype
    assert actual_name == name


@pytest.mark.parametrize(
    ('coordinates', 'expected_message'),
    [
        (_ONE_DIMENSIONAL_INVALID_COORDINATES, r'must have shape \(N, 2\)'),
        (_WRONG_COLUMN_COUNT_COORDINATES, r'must have shape \(N, 2\)'),
        (_TOO_FEW_COORDINATES, 'at least 5 points'),
        (_INTEGER_COORDINATES, 'real floating-point dtype'),
        (_COMPLEX_COORDINATES, 'real floating-point dtype'),
        (_STRING_COORDINATES, 'real floating-point dtype'),
        (_NON_FINITE_COORDINATES, 'must be finite'),
        (_INFINITE_COORDINATES, 'must be finite'),
        (_DUPLICATE_COORDINATES, 'not contain duplicate points'),
        (_CLOCKWISE_COORDINATES, 'ordered counter-clockwise'),
        (_COLLINEAR_COORDINATES, 'ordered counter-clockwise'),
    ],
    ids=[
        'one_dimensional_shape',
        'wrong_column_count',
        'too_few_coordinates',
        'integer_coordinates',
        'complex_coordinates',
        'string_coordinates',
        'non_finite_coordinates',
        'infinite_coordinates',
        'duplicate_coordinates',
        'clockwise_order',
        'zero_signed_area',
    ],
)
def test_airfoil_construction_rejects_invalid_coordinates(
    coordinates: np.ndarray,
    expected_message: str,
) -> None:
    # GIVEN: invalid coordinates and the expected constructor error message.

    name = _CONSTRUCTION_NAME

    # WHEN: the airfoil constructor validates the coordinates.

    with pytest.raises(
        ValueError,
        match=expected_message,
    ) as actual_exception:
        Airfoil(coordinates, name)

    # THEN: the actual exception contains the expected message.

    assert actual_exception.type is ValueError


def test_computation_of_airfoil_geometric_properties() -> None:
    # GIVEN: valid coordinates and named expected tangent, normal, and length arrays.

    coordinates = _valid_coordinates()

    # WHEN: geometric properties are computed.

    airfoil = Airfoil(coordinates, _CONSTRUCTION_NAME)
    actual_tangents = airfoil.tangents()
    actual_normals = airfoil.normals()
    actual_lengths = airfoil.lengths()

    # THEN: the actual properties equal the named expected arrays.

    np.testing.assert_allclose(actual_tangents, _EXPECTED_TANGENTS)
    np.testing.assert_allclose(actual_normals, _EXPECTED_NORMALS)
    np.testing.assert_allclose(actual_lengths, _EXPECTED_LENGTHS)


def test_chord_alignment_transform_maps_leading_and_trailing_edges_to_unit_chord() -> None:
    # GIVEN: valid coordinates transformed by known rotation, scale, and translation.

    coordinates = _valid_coordinates()
    forward_rotation = np.array(
        [
            [
                np.cos(_FORWARD_ROTATION_ANGLE),
                -np.sin(_FORWARD_ROTATION_ANGLE),
            ],
            [
                np.sin(_FORWARD_ROTATION_ANGLE),
                np.cos(_FORWARD_ROTATION_ANGLE),
            ],
        ],
    )
    transformed_coordinates = (
        _FORWARD_SCALE * (coordinates @ forward_rotation.T) + _FORWARD_TRANSLATION
    )

    # WHEN: an airfoil is constructed from the transformed coordinates.

    airfoil = Airfoil(transformed_coordinates)
    actual_rotation, actual_translation, actual_scale = airfoil.chord_alignment_transform()
    aligned_coordinates = (
        actual_scale
        * (actual_rotation @ transformed_coordinates.T + actual_translation[:, np.newaxis])
    ).T
    trailing_edge = (transformed_coordinates[0] + transformed_coordinates[-1]) / 2
    leading_edge_index = np.argmax(
        np.linalg.norm(transformed_coordinates - trailing_edge, axis=1),
    )

    # THEN: the actual alignment maps the leading edge and trailing midpoint to unit chord.

    np.testing.assert_allclose(
        aligned_coordinates[leading_edge_index],
        _ORIGIN,
        atol=_ALIGNMENT_TOLERANCE,
    )
    np.testing.assert_allclose(
        (aligned_coordinates[0] + aligned_coordinates[-1]) / 2,
        _UNIT_TRAILING_POINT,
        atol=_ALIGNMENT_TOLERANCE,
    )
    np.testing.assert_allclose(
        actual_rotation @ actual_rotation.T,
        _TWO_DIMENSIONAL_IDENTITY,
        atol=_ALIGNMENT_TOLERANCE,
    )
    assert actual_scale > 0


def test_update_airfoil_coordinates() -> None:
    # GIVEN: an airfoil and scaled replacement coordinates.

    initial_coordinates = _valid_coordinates()
    expected_coordinates = _COORDINATE_SCALE * initial_coordinates
    airfoil = Airfoil(initial_coordinates, _CONSTRUCTION_NAME)

    # WHEN: the coordinates are updated.

    airfoil.update(expected_coordinates)
    actual_coordinates = airfoil.coordinates

    # THEN: the actual coordinates equal the scaled replacement coordinates.

    np.testing.assert_array_equal(actual_coordinates, expected_coordinates)


@pytest.mark.parametrize(
    ('updated_coordinates', 'expected_message'),
    [
        (
            _ONE_DIMENSIONAL_INVALID_COORDINATES,
            r'must have shape \(N, 2\)',
        ),
        (
            _WRONG_COLUMN_COUNT_COORDINATES,
            r'must have shape \(N, 2\)',
        ),
        (
            _valid_coordinates()[:-2],
            'at least 5 points',
        ),
        (_INTEGER_COORDINATES, 'real floating-point dtype'),
        (_COMPLEX_COORDINATES, 'real floating-point dtype'),
        (_STRING_COORDINATES, 'real floating-point dtype'),
        (_NON_FINITE_COORDINATES, 'must be finite'),
        (_INFINITE_COORDINATES, 'must be finite'),
        (_DUPLICATE_COORDINATES, 'not contain duplicate points'),
        (_CLOCKWISE_COORDINATES, 'ordered counter-clockwise'),
        (_COLLINEAR_COORDINATES, 'ordered counter-clockwise'),
        (
            _valid_coordinates()[:-1],
            'Cannot change number of coordinates',
        ),
    ],
    ids=[
        'one_dimensional_shape',
        'wrong_column_count',
        'too_few_coordinates',
        'integer_coordinates',
        'complex_coordinates',
        'string_coordinates',
        'non_finite_coordinates',
        'infinite_coordinates',
        'duplicate_coordinates',
        'clockwise_order',
        'zero_signed_area',
        'different_number_of_coordinates',
    ],
)
def test_update_airfoil_coordinates_rejects_invalid_coordinates(
    updated_coordinates: np.ndarray,
    expected_message: str,
) -> None:
    # GIVEN: an airfoil with valid coordinates and invalid replacement coordinates.

    initial_coordinates = _valid_coordinates()
    airfoil = Airfoil(initial_coordinates, _CONSTRUCTION_NAME)

    # WHEN: the airfoil update validates the replacement coordinates.

    with pytest.raises(
        ValueError,
        match=expected_message,
    ) as actual_exception:
        airfoil.update(updated_coordinates)
    actual_coordinates = airfoil.coordinates

    # THEN: the actual exception matches the expected message and coordinates remain unchanged.

    assert actual_exception.type is ValueError
    np.testing.assert_array_equal(actual_coordinates, initial_coordinates)


def test_geometry_is_recomputed_after_coordinates_change() -> None:
    # GIVEN: an initial airfoil and an updated coordinate array.

    initial_coordinates = _valid_coordinates()
    updated_coordinates = initial_coordinates.copy()
    updated_coordinates[-1, 1] = _UPDATED_TRAILING_Y
    airfoil = Airfoil(initial_coordinates)

    # WHEN: the airfoil coordinates are updated.

    airfoil.update(updated_coordinates)
    baseline_tangents = Airfoil(initial_coordinates).tangents()
    baseline_normals = Airfoil(initial_coordinates).normals()
    baseline_lengths = Airfoil(initial_coordinates).lengths()
    actual_tangents = airfoil.tangents()
    actual_normals = airfoil.normals()
    actual_lengths = airfoil.lengths()

    # THEN: the actual updated geometry differs from geometry for the initial coordinates.

    assert not np.array_equal(actual_tangents, baseline_tangents)
    assert not np.array_equal(actual_normals, baseline_normals)
    assert not np.array_equal(actual_lengths, baseline_lengths)


def test_airfoil_uses_default_name() -> None:
    # GIVEN: valid coordinates and the named default name.

    coordinates = _valid_coordinates()
    expected_name = 'UnnamedAirfoil'

    # WHEN: an airfoil is constructed without a name.

    airfoil = Airfoil(coordinates)
    actual_name = airfoil.name

    # THEN: the actual name equals the named default.

    assert actual_name == expected_name


def test_airfoil_representation() -> None:
    # GIVEN: a valid named airfoil and expected representation fragments.

    coordinates = _valid_coordinates()
    name = _CONSTRUCTION_NAME
    airfoil = Airfoil(coordinates, name)
    expected_coordinate_text = str(coordinates)

    # WHEN: the airfoil is converted to a string.

    actual_representation = repr(airfoil)

    # THEN: the actual representation contains the name and coordinate text.

    assert name in actual_representation
    assert expected_coordinate_text in actual_representation


def test_close_polygon_appends_first_coordinate_when_open() -> None:
    # GIVEN: an open polygon.

    coordinates = _valid_coordinates()

    # WHEN: the polygon is closed.

    closed_coordinates = close_polygon(coordinates)

    # THEN: the actual array repeats the first coordinate at the end.

    assert closed_coordinates is not None
    expected_coordinates = np.concatenate(
        (coordinates, coordinates[:1]),
        axis=0,
    )
    np.testing.assert_array_equal(closed_coordinates, expected_coordinates)


def test_close_polygon_does_nothing_when_closed() -> None:
    # GIVEN: a closed polygon.

    coordinates = np.concatenate(
        (_valid_coordinates(), _valid_coordinates()[:1]),
        axis=0,
    )

    # WHEN: the polygon closure helper is applied.

    actual_coordinates = close_polygon(coordinates)

    # THEN: no closed array is returned.

    assert actual_coordinates is None


@pytest.mark.parametrize(
    'coordinates',
    [
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    ],
    ids=['one_coordinate', 'two_coordinates'],
)
def test_close_polygon_does_nothing_below_minimum_size(
    coordinates: np.ndarray,
) -> None:
    # GIVEN: a polygon below the closing helper's minimum size.
    # WHEN: the polygon closure helper is applied.

    actual_coordinates = close_polygon(coordinates)

    # THEN: no closed array is returned.

    assert actual_coordinates is None


def test_open_polygon_removes_repeated_final_coordinate_when_closed() -> None:
    # GIVEN: a closed polygon.

    coordinates = np.concatenate(
        (_valid_coordinates(), _valid_coordinates()[:1]),
        axis=0,
    )

    # WHEN: the polygon is opened.

    actual_coordinates = open_polygon(coordinates)

    # THEN: the actual first and final coordinates are no longer equal.

    assert actual_coordinates is not None
    assert not np.array_equal(actual_coordinates[0], actual_coordinates[-1])


def test_open_polygon_does_nothing_when_open() -> None:
    # GIVEN: an open polygon.

    coordinates = _valid_coordinates()

    # WHEN: the polygon opening helper is applied.

    actual_coordinates = open_polygon(coordinates)

    # THEN: no open array is returned.

    assert actual_coordinates is None


@pytest.mark.parametrize(
    'coordinates',
    [
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
    ],
    ids=['one_coordinate', 'three_coordinates'],
)
def test_open_polygon_does_nothing_below_minimum_size(
    coordinates: np.ndarray,
) -> None:
    # GIVEN: a polygon below the opening helper's minimum size.
    # WHEN: the polygon opening helper is applied.

    actual_coordinates = open_polygon(coordinates)

    # THEN: no open array is returned.

    assert actual_coordinates is None
