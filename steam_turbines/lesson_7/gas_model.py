from typing import Union, Optional

import numpy as np
from iapws.iapws97 import IAPWS97

from utils import preprocess_args

MPa = 1e6

default_types = Union[int, float, np.ndarray]


def get_p_from_h_s(h: default_types, s: default_types) -> np.ndarray:
    h, s = preprocess_args(h, s)
    p = np.array([IAPWS97(h=_h / 1000, s=_s).P * MPa for _h, _s in zip(h, s)])
    return p


def get_v_from_h_s(h: default_types, s: default_types) -> np.ndarray:
    h, s = preprocess_args(h, s)
    v = np.array([IAPWS97(h=_h / 1000, s=_s).v for _h, _s in zip(h, s)])
    return v


def get_t_from_h_s(h: default_types, s: default_types) -> np.ndarray:
    h, s = preprocess_args(h, s)
    t = np.array([IAPWS97(h=_h / 1000, s=_s).T for _h, _s in zip(h, s)])
    return t


def get_h_from_p_s(p: default_types, s: default_types) -> np.ndarray:
    p, s = preprocess_args(p, s)
    h = np.array([IAPWS97(P=_p / MPa, s=_s).h * 1000 for _p, _s in zip(p, s)])
    return h


def get_h_from_p_t(p: default_types, t: default_types) -> np.ndarray:
    p, t = preprocess_args(p, t)
    h = np.array([IAPWS97(P=_p / MPa, T=_t).h * 1000 for _p, _t in zip(p, t)])
    return h


def get_s_from_p_h(p: default_types, h: default_types) -> np.ndarray:
    p, h = preprocess_args(p, h)
    s = np.array([IAPWS97(P=_p / MPa, h=_h / 1000).s for _p, _h in zip(p, h)])
    return s


def get_s_from_p_t(p: default_types, t: default_types) -> np.ndarray:
    p, t = preprocess_args(p, t)
    s = np.array([IAPWS97(P=_p / MPa, T=_t).s for _p, _t in zip(p, t)])
    return s


def get_a_from_h_s(h: default_types, s: default_types) -> np.ndarray:
    h, s = preprocess_args(h, s)
    a = np.array([IAPWS97(h=_h / 1000, s=_s).w for _h, _s in zip(h, s)])
    return a


def get_p_from_state(
        h: Optional[default_types] = None,
        s: Optional[default_types] = None,
) -> np.ndarray:
    if h is not None and s is not None:
        return get_p_from_h_s(h, s)

    raise KeyError("Wrong arguments combination!")


def get_v_from_state(
        h: Optional[default_types] = None,
        s: Optional[default_types] = None,
) -> np.ndarray:
    if h is not None and s is not None:
        return get_v_from_h_s(h, s)

    raise KeyError("Wrong arguments combination!")


def get_t_from_state(
        h: Optional[default_types] = None,
        s: Optional[default_types] = None,
) -> np.ndarray:
    if h is not None and s is not None:
        return get_t_from_h_s(h, s)

    raise KeyError("Wrong arguments combination!")


def get_s_from_state(
        p: Optional[default_types] = None,
        h: Optional[default_types] = None,
        t: Optional[default_types] = None,
) -> np.ndarray:
    if p is not None and h is not None:
        return get_s_from_p_h(p, h)
    if p is not None and t is not None:
        return get_s_from_p_t(p, t)

    raise KeyError("Wrong arguments combination!")


def get_h_from_state(
        p: Optional[default_types] = None,
        s: Optional[default_types] = None,
        t: Optional[default_types] = None,
) -> np.ndarray:
    if p is not None and s is not None:
        return get_h_from_p_s(p, s)
    if p is not None and t is not None:
        return get_h_from_p_t(p, t)

    raise KeyError("Wrong arguments combination!")


def get_sound_speed_from_state(
        h: Optional[default_types] = None,
        s: Optional[default_types] = None,
) -> np.ndarray:
    if h is not None and s is not None:
        return get_a_from_h_s(h, s)

    raise KeyError("Wrong arguments combination!")
