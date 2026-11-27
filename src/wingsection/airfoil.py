from typing import ClassVar

import numpy as np

from wingsection.types import Matrix, Vector


class Airfoil[T: np.floating]:
    """Represent a wing cross section by an ordered sequence of surface coordinates.

    The coordinate sequence must contain finite, unique, real floating-point
    values with at least five points. Counter-clockwise orientation is determined
    by positive signed area. Self-intersections are not checked.

    Parameters
    ----------
    coordinates
        Two-dimensional real floating-point array representing surface points.
    name
        Human-readable name used to identify the airfoil.

    Attributes
    ----------
    coordinates
        Copy of the current two-dimensional coordinate array.
    name
        Human-readable name used to identify the airfoil.

    Raises
    ------
    ValueError
        Raised when coordinates do not meet the shape, count, dtype,
        finiteness, uniqueness, or orientation requirements.
    """

    _COORDINATE_DIMENSIONS: ClassVar[int] = 2
    _MINIMUM_COORDINATE_COUNT: ClassVar[int] = 5

    def __init__(self, coordinates: Matrix[T], name: str = 'UnnamedAirfoil') -> None:
        """Construct an airfoil from a set of coordinates.

        Copy the validated coordinate array so later changes to the caller's
        array cannot modify the stored airfoil.

        Parameters
        ----------
        coordinates
            A 2D real floating-point array of shape ``(N, 2)`` representing the
            x and y coordinates of the airfoil surface.
        name
            The name of the airfoil.

        Raises
        ------
        ValueError
            If the coordinates do not meet the shape, count, dtype, finiteness,
            uniqueness, and orientation requirements.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> Airfoil(coordinates).name
        'UnnamedAirfoil'
        """
        self._validate_coordinates(coordinates)

        self._coordinates = coordinates.copy()
        self._name = name

    def __repr__(self) -> str:
        """Return a concise representation of the airfoil.

        The representation includes the configured name and an array copy of the
        current coordinates for diagnostics and interactive display.

        Returns
        -------
        representation
            String representation of the airfoil name and coordinates.
        """
        return f'Airfoil(name={self._name}, coordinates={self._coordinates})'

    def update(self, coordinates: Matrix[T]) -> None:
        """Replace the airfoil coordinates.

        Validate the replacement before modifying the stored array. The
        replacement must preserve the existing coordinate count and orientation.

        Parameters
        ----------
        coordinates
            A 2D real floating-point array of shape ``(N, 2)`` representing the
            replacement coordinates. The number of points must match the current
            coordinates.

        Raises
        ------
        ValueError
            If the new coordinates are invalid or their count differs from the current coordinates.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> airfoil = Airfoil(coordinates)
        >>> airfoil.update(2 * coordinates)
        >>> airfoil.coordinates.shape
        (5, 2)
        """
        self._validate_coordinates(coordinates)

        if self._coordinates.shape != coordinates.shape:
            message = (
                f'Cannot change number of coordinates: '
                f'{self._coordinates.shape} != {coordinates.shape}.'
            )
            raise ValueError(message)

        self._coordinates[...] = coordinates

    def lengths(self) -> Vector[T]:
        """Compute lengths between consecutive coordinates.

        The closing edge from the last point back to the first is not included.
        The result uses the Euclidean norm of each coordinate difference.

        Returns
        -------
        lengths
            A 1D array of shape ``(N-1,)`` containing lengths between consecutive
            points of the airfoil surface.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> Airfoil(coordinates).lengths().shape
        (4,)
        """
        differences = np.diff(self._coordinates, axis=0)
        lengths = np.linalg.norm(differences, axis=1)

        return lengths

    def chord_alignment_transform(
        self,
    ) -> tuple[Matrix[T], Vector[T], T]:
        """Compute a transform that aligns and normalizes the airfoil chord.

        The trailing edge is taken as the midpoint of the first and last
        coordinates. The leading edge is the coordinate farthest from that
        midpoint. This assumes the coordinate sequence starts and ends on the
        trailing edge.

        Apply the returned values to column-vector coordinates ``p`` as
        ``scale * (rotation @ p + translation)``. The leading edge maps to
        ``(0, 0)`` and the trailing-edge midpoint maps to ``(1, 0)``.

        Returns
        -------
        rotation
            Two-dimensional rotation matrix that orients the chord.
        translation
            Translation vector that moves the leading edge to the origin.
        scale
            Uniform scale factor that maps the chord length to one.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> rotation, translation, scale = Airfoil(coordinates).chord_alignment_transform()
        >>> (rotation.shape, translation.shape, bool(scale > 0))
        ((2, 2), (2,), True)
        """
        coordinates = self._coordinates

        trailing_edge = (coordinates[0] + coordinates[-1]) / 2
        leading_edge_index = np.argmax(
            np.linalg.norm(coordinates - trailing_edge, axis=1),
        )
        leading_edge = coordinates[leading_edge_index]

        chord = trailing_edge - leading_edge
        chord_length = np.linalg.norm(chord)

        cosine = chord[0] / chord_length
        sine = chord[1] / chord_length

        rotation = np.array([[cosine, sine], [-sine, cosine]])
        translation = -(rotation @ leading_edge)
        scale = 1 / chord_length

        return rotation, translation, scale

    def normals(self) -> Matrix[T]:
        """Compute unit outward normals at each airfoil coordinate.

        Each normal is the clockwise 90-degree rotation of its unit tangent, which
        points outward for the required counter-clockwise coordinate ordering.

        Returns
        -------
        normals
            A 2D array of shape ``(N, 2)`` containing unit outward normals.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> Airfoil(coordinates).normals().shape
        (5, 2)
        """
        tangents = self.tangents()

        normals = np.empty_like(tangents)
        normals[:, 0] = tangents[:, 1]
        normals[:, 1] = -tangents[:, 0]

        return normals

    def tangents(self) -> Matrix[T]:
        """Compute unit tangents at each airfoil coordinate.

        Tangents are estimated with finite differences along the coordinate
        sequence: centered differences at interior points and one-sided
        differences at the endpoints. The differences are normalized to unit
        length.

        Returns
        -------
        tangents
            A 2D array of shape ``(N, 2)`` containing unit tangent vectors.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> Airfoil(coordinates).tangents().shape
        (5, 2)
        """
        tangents = np.gradient(self._coordinates, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        return tangents

    @property
    def coordinates(self) -> Matrix[T]:
        """Get a copy of the airfoil coordinates.

        Use the ``update`` method to modify coordinates. Each access produces an
        independent copy so direct caller mutations cannot change this object.

        Returns
        -------
        coordinates
            A copy of shape ``(N, 2)`` containing the x and y coordinates of the
            airfoil surface.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> airfoil = Airfoil(coordinates)
        >>> airfoil.coordinates is coordinates
        False
        """
        return self._coordinates.copy()

    @property
    def name(self) -> str:
        """Get the name of the airfoil.

        The name remains a descriptive string supplied at construction or left at
        the package's default value.

        Returns
        -------
        name
            The name of the airfoil.

        Examples
        --------
        >>> coordinates = np.array([[1.0, 0.6], [0.25, 0.7], [0.0, 0.5], [0.25, 0.3], [1.0, 0.4]])
        >>> Airfoil(coordinates).name
        'UnnamedAirfoil'
        """
        return self._name

    def _validate_coordinates(self, coordinates: Matrix[T]) -> None:
        """Validate airfoil coordinates.

        Check shape, count, dtype, finiteness, uniqueness, and signed orientation
        before either construction or coordinate updates proceed.

        Parameters
        ----------
        coordinates
            A 2D real floating-point array of shape ``(N, 2)``.

        Raises
        ------
        ValueError
            If coordinates have an invalid shape, too few points, a non-real or
            non-floating dtype, non-finite values, duplicates, or non-positive
            signed area.
        """
        if (
            coordinates.ndim != Airfoil._COORDINATE_DIMENSIONS
            or coordinates.shape[1] != Airfoil._COORDINATE_DIMENSIONS
        ):
            message = (
                f'Coordinates must have shape (N, {Airfoil._COORDINATE_DIMENSIONS});'
                f' got {coordinates.shape}.'
            )
            raise ValueError(message)

        if coordinates.shape[0] < Airfoil._MINIMUM_COORDINATE_COUNT:
            message = (
                f'Coordinates must contain at least {Airfoil._MINIMUM_COORDINATE_COUNT} points;'
                f' got {coordinates.shape[0]}.'
            )
            raise ValueError(message)

        if not np.issubdtype(coordinates.dtype, np.floating):
            message = 'Coordinates must use a real floating-point dtype.'
            raise ValueError(message)

        if not np.all(np.isfinite(coordinates)):
            message = 'All coordinates must be finite.'
            raise ValueError(message)

        unique_coordinate_count = len(np.unique(coordinates, axis=0))
        if unique_coordinate_count != coordinates.shape[0]:
            message = (
                f'Coordinates must not contain duplicate points;'
                f' got {coordinates.shape[0] - unique_coordinate_count} duplicates.'
            )
            raise ValueError(message)

        if not self._is_counter_clockwise(coordinates):
            message = 'Coordinates must be ordered counter-clockwise.'
            raise ValueError(message)

    def _is_counter_clockwise(self, coordinates: Matrix[T]) -> bool:
        """Check whether the closed coordinate sequence has positive signed area.

        The sequence is treated as closed by joining its last point to its first
        if necessary. The signed area is computed with the shoelace formula.

        Parameters
        ----------
        coordinates
            A 2D array of shape ``(N, 2)`` representing the x and y coordinates
            of the airfoil surface.

        Returns
        -------
        counter_clockwise
            Whether the signed area is positive.
        """
        closed_polygon = close_polygon(coordinates)
        if closed_polygon is not None:
            coordinates = closed_polygon

        x = coordinates[:, 0]
        y = coordinates[:, 1]
        double_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

        return bool(double_area > 0)


def close_polygon[T: np.floating](coordinates: Matrix[T]) -> Matrix[T] | None:
    """Close a polygon by appending its first point when needed.

    Return a copy for an open polygon with at least three rows and leave a
    closed polygon unchanged. This helper performs no shape or geometry validation.

    Parameters
    ----------
    coordinates
        A 2D array of shape ``(N, 2)`` representing the x and y coordinates
        of a polygon.

    Returns
    -------
    closed_coordinates
        A new array with the first point appended, or ``None`` if the input is
        already closed or has fewer than three points.

    Examples
    --------
    >>> coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> closed = close_polygon(coordinates)
    >>> closed.shape
    (4, 2)
    """
    minimum_polygon_size = 3
    if coordinates.shape[0] >= minimum_polygon_size and not np.array_equal(
        coordinates[0],
        coordinates[-1],
    ):
        return np.concatenate((coordinates, coordinates[:1]), axis=0)

    return None


def open_polygon[T: np.floating](coordinates: Matrix[T]) -> Matrix[T] | None:
    """Open a closed polygon by removing its repeated final point.

    Return a copy for a closed polygon with at least four rows and leave an
    open polygon unchanged. This helper performs no shape or geometry validation.

    Parameters
    ----------
    coordinates
        A 2D array of shape ``(N, 2)`` representing the x and y coordinates
        of a polygon.

    Returns
    -------
    open_coordinates
        A copy without the final point, or ``None`` if the input is not closed or
        has fewer than four points.

    Examples
    --------
    >>> coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    >>> open_coordinates = open_polygon(coordinates)
    >>> open_coordinates.shape
    (3, 2)
    """
    minimum_closed_polygon_size = 4
    if coordinates.shape[0] >= minimum_closed_polygon_size and np.array_equal(
        coordinates[0],
        coordinates[-1],
    ):
        return coordinates[:-1].copy()

    return None
