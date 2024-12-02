from typing import Union, Tuple
import numpy as np
from gas_model import (
    get_p_from_state,
    get_v_from_state,
    get_s_from_state,
    get_h_from_state,
    get_t_from_state
)

default_types = Union[int, float, np.ndarray]


def compute_heat_drops(
    p0: default_types,
    t0: default_types,
    p2: default_types,
    dor: default_types,
    inlet_speed: default_types,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inlet_energy = (inlet_speed ** 2) / 2
    s0 = get_s_from_state(p=p0, t=t0)
    _h0 = get_h_from_state(p=p0, s=s0)
    h0 = _h0 + inlet_energy
    h2t = get_h_from_state(p=p2, s=s0)

    total_heat_drop = h0 - h2t
    stator_heat_drop = (1 - dor) * total_heat_drop
    rotor_heat_drop = total_heat_drop - stator_heat_drop
    return total_heat_drop, stator_heat_drop, rotor_heat_drop


def compute_intermedia_point(
    h_start: np.ndarray,
    s_start: np.ndarray,
    dh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_end = h_start - dh
    p_end = get_p_from_state(h=h_end, s=s_start)
    v_end = get_v_from_state(h=h_end, s=s_start)
    t_end = get_t_from_state(h=h_end, s=s_start)
    return p_end, v_end, t_end, h_end


def compute_triangle(
        u: float,
        inlet_speed: default_types,
        inlet_angle: default_types
) -> Tuple[default_types, default_types]:
    dependent_speed = (inlet_speed**2 + u**2 - 2 * inlet_speed * u * np.cos(inlet_angle)) ** 0.5
    outlet_angle = np.arccos((inlet_speed * np.cos(inlet_angle) - u) / dependent_speed)
    return dependent_speed, outlet_angle


def compute_speed_coefficient(blade_length, chord, is_rotor) -> float:
    if is_rotor:
        return 0.96 - 0.014 * chord / blade_length
    return 0.98 - 0.008 * chord / blade_length


def compute_discharge_coefficient(blade_length, chord, is_rotor) -> float:
    if is_rotor:
        return 0.965 - 0.010 * chord / blade_length
    return 0.982 - 0.005 * chord / blade_length


def compute_stator_blade_length(mass_flow_rate, d, c1t, v1t, alpha_1, chord):
    is_solved = False
    nu = 1
    nu_new = None
    blade_length = 0
    while not is_solved:
        if nu_new is not None:
            nu = nu_new
        blade_length = mass_flow_rate * v1t / ((np.sin(alpha_1) * np.pi * d) * c1t * nu)
        nu_new = compute_discharge_coefficient(blade_length, chord, is_rotor=False)
        if np.isclose(nu, nu_new).all():
            is_solved = True
    return blade_length


def compute_beta_2(mass_flow_rate, d, w2t, v2t, l2, nu):
    sin_beta_2 = mass_flow_rate * v2t / (nu * w2t * np.pi * d * l2)
    return np.arcsin(sin_beta_2)


def move_angle(angle, speed_coefficient, discharge_coefficient):
    return np.arcsin(discharge_coefficient * np.sin(angle) / speed_coefficient)
