from typing import Any, Union
import numpy as np


def check_same_size(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Size of arrays should be the same! a.shape = {a.shape}, b.shape = {b.shape}")


def check_valid_types(a: Any) -> None:
    if not isinstance(a, (int, float, np.ndarray)):
        raise ValueError(f"Variable have type {type(a)} that is not allowed!")


def uproll_var(a: Union[int, float], a_like: np.ndarray) -> np.ndarray:
    return np.ones_like(a_like) * a


def preprocess_args(*args) -> Any:
    is_digits = dict()
    is_arrays = dict()
    for i, val in enumerate(args):
        check_valid_types(val)
        if isinstance(val, (int, float)):
            is_digits[i] = val
        else:
            is_arrays[i] = val
    if len(is_arrays) > 1:
        vals = list(is_arrays.values())
        for left, right in zip(vals[:-1], vals[1:]):
            check_same_size(left, right)
    if len(is_arrays) == 0:
        return [np.array([val]) for val in args]
    a_like = list(is_arrays.values())[0]
    for key, values in is_digits.items():
        is_digits[key] = uproll_var(a=values, a_like=a_like)
    is_digits.update(is_arrays)
    to_return = [is_digits[i] for i in range(len(args))]
    return to_return
