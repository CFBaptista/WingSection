import numpy as np

Scalar = np.floating
type Vector[T: Scalar] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: Scalar] = np.ndarray[tuple[int, int], np.dtype[T]]
type Tensor[T: Scalar] = Scalar | Vector[T] | Matrix[T]
