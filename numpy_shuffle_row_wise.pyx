import numpy as np
cimport cython

from cython.parallel import prange


# import numpy as np
import random

DTYPE = np.intc

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def compute(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):

    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    # array_1.shape is now a C array, no it's not possible
    # to compare it simply by using == without a for-loop.
    # To be able to compare it to array_2.shape easily,
    # we convert them both to Python tuples.
    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

from libc.stdlib cimport rand, RAND_MAX

@cython.cdivision(True)
cdef inline int randint(int min_n, int max_n):
    return rand() % (max_n - min_n + 1) + min_n


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def shuffle_row_wise(long[:, :] a):
    cdef Py_ssize_t n_rows = a.shape[0]
    cdef Py_ssize_t n_vals = a.shape[1]
    # assert a.dtype == np.intc

    cdef Py_ssize_t max_index = n_vals - 1
    cdef long tmp
    cdef Py_ssize_t row_n, i, j
    for row_n in range(n_rows):
        for i in range(max_index):
            j = rand() % (max_index - i + 1) + i
            tmp = a[row_n, i]
            a[row_n, i] = a[row_n, j]
            a[row_n, j] = tmp


    # def shuffle(self, object x):
    #     cdef:
    #         npy_intp i, j, n = len(x), stride, itemsize
    #         char* x_ptr
    #         char* buf_ptr

    #     if type(x) is np.ndarray and x.ndim == 1 and x.size:
    #         # Fast, statically typed path: shuffle the underlying buffer.
    #         # Only for non-empty, 1d objects of class ndarray (subclasses such
    #         # as MaskedArrays may not support this approach).
    #         x_ptr = <char*><size_t>x.ctypes.data
    #         stride = x.strides[0]
    #         itemsize = x.dtype.itemsize
    #         # As the array x could contain python objects we use a buffer
    #         # of bytes for the swaps to avoid leaving one of the objects
    #         # within the buffer and erroneously decrementing it's refcount
    #         # when the function exits.
    #         buf = np.empty(itemsize, dtype=np.int8) # GC'd at function exit
    #         buf_ptr = <char*><size_t>buf.ctypes.data
    #         with self.lock:
    #             # We trick gcc into providing a specialized implementation for
    #             # the most common case, yielding a ~33% performance improvement.
    #             # Note that apparently, only one branch can ever be specialized.
    #             if itemsize == sizeof(npy_intp):
    #                 self._shuffle_raw(n, sizeof(npy_intp), stride, x_ptr, buf_ptr)
    #             else:
    #                 self._shuffle_raw(n, itemsize, stride, x_ptr, buf_ptr)
    #     elif isinstance(x, np.ndarray) and x.ndim and x.size:
    #         buf = np.empty_like(x[0,...])
    #         with self.lock:
    #             for i in reversed(range(1, n)):
    #                 j = rk_interval(i, self.internal_state)
    #                 buf[...] = x[j]
    #                 x[j] = x[i]
    #                 x[i] = buf
    #     else:
    #         # Untyped path.
    #         with self.lock:
    #             for i in reversed(range(1, n)):
    #                 j = rk_interval(i, self.internal_state)
    #                 x[i], x[j] = x[j], x[i]

    # cdef inline _shuffle_raw(self, npy_intp n, npy_intp itemsize,
    #                          npy_intp stride, char* data, char* buf):
    #     cdef npy_intp i, j
    #     for i in reversed(range(1, n)):
    #         j = rk_interval(i, self.internal_state)
    #         if i == j : continue # i == j is not needed and memcpy is undefined.
    #         string.memcpy(buf, data + j * stride, itemsize)
    #         string.memcpy(data + j * stride, data + i * stride, itemsize)
    #         string.memcpy(data + i * stride, buf, itemsize)