from typing import Optional, Union
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import turbine_functions as F
from gas_model import (
    get_s_from_state,
    get_h_from_state,
    get_sound_speed_from_state,
)

default_types = Union[int, float, np.ndarray]


class Triangles:
    u: default_types
    c1: default_types
    c2: default_types
    w1: default_types
    w2: default_types
    alpha_1: default_types
    alpha_2: default_types
    beta_1: default_types
    beta_2: default_types

    def __init__(self) -> None:
        pass

    def _construct_vectors(self):
        zero = np.zeros_like(self.c1)
        cos_alpha_1 = np.cos(self.alpha_1)
        sin_alpha_1 = np.sin(self.alpha_1)
        cos_beta_2 = np.cos(self.beta_2)
        sin_beta_2 = np.sin(self.beta_2)
        u = self.u if isinstance(self.u, np.ndarray) else np.ones_like(self.c1) * self.u

        c1_plot = [np.vstack([zero, -self.c1 * cos_alpha_1]).T, np.vstack([zero, -self.c1 * sin_alpha_1]).T]
        u1_plot = [
            np.vstack([-self.c1 * cos_alpha_1, -self.c1 * cos_alpha_1 + u]).T,
            np.vstack([-self.c1 * sin_alpha_1, -self.c1 * sin_alpha_1]).T
        ]
        w1_plot = [np.vstack([zero, -self.c1 * cos_alpha_1 + u]).T, np.vstack([zero, -self.c1 * sin_alpha_1]).T]

        w2_plot = [np.vstack([zero, self.w2 * cos_beta_2]).T, np.vstack([zero, -self.w2 * sin_beta_2]).T]
        u2_plot = [
            np.vstack([self.w2 * cos_beta_2, self.w2 * cos_beta_2 - u]).T,
            np.vstack([-self.w2 * sin_beta_2, -self.w2 * sin_beta_2]).T
        ]
        c2_plot = [
            np.vstack([zero, self.w2 * cos_beta_2 - u]).T,
            np.vstack([zero, -self.w2 * sin_beta_2]).T
        ]

        return c1_plot, w1_plot, u1_plot, c2_plot, w2_plot, u2_plot

    def plot(self, ax: Optional[Axes] = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        c1, w1, u1, c2, w2, u2 = self._construct_vectors()
        for i in range(self.c1.shape[0]):
            ax.plot(c1[0][i], c1[1][i], label='C1', c='red')
            ax.plot(u1[0][i], u1[1][i], label='u1', c='blue')
            ax.plot(w1[0][i], w1[1][i], label='W1', c='green')

            ax.plot(w2[0][i], w2[1][i], label='W2', c='green')
            ax.plot(u2[0][i], u2[1][i], label='u2', c='blue')
            ax.plot(c2[0][i], c2[1][i], label='C2', c='red')

        ax.set_title("Треугольник скоростей")


class TurbineStage:
    u_div_c: default_types
    blade_efficiency: default_types
    blade_power: default_types

    def __init__(
            self,
            mass_flow_rate: default_types,
            average_diameter: default_types,
            inlet_pressure: default_types,
            inlet_temperature: default_types,
            outlet_pressure: default_types,
            rotation_speed: default_types,
            degree_of_reaction: default_types,
            stator_outlet_angle: default_types,
            overlapping: default_types,
            inlet_speed: default_types = 0,
            is_last: bool = True,
    ) -> None:
        self.triangles = Triangles()
        self.mass_flow_rate = mass_flow_rate
        self.average_diameter = average_diameter
        self.inlet_pressure = inlet_pressure
        self.inlet_temperature = inlet_temperature
        self.outlet_pressure = outlet_pressure
        self.rotation_speed = rotation_speed
        self.degree_of_reaction = degree_of_reaction
        self.stator_outlet_angle = stator_outlet_angle
        self.overlapping = overlapping
        self.inlet_speed = inlet_speed
        self.is_last = is_last

        self.blade_efficiency, self.u_div_c, self.blade_power = self.design()

    def design(self):
        chord = 50 / 1000

        self.triangles.u = np.pi * self.rotation_speed * self.average_diameter
        total_heat_drop, stator_heat_drop, rotor_heat_drop = F.compute_heat_drops(
            p0=self.inlet_pressure,
            t0=self.inlet_temperature,
            p2=self.outlet_pressure,
            dor=self.degree_of_reaction,
            inlet_speed=self.inlet_speed
        )
        dummy_speed = (2 * total_heat_drop) ** 0.5
        u_div_c = self.triangles.u / dummy_speed

        h0 = get_h_from_state(p=self.inlet_pressure, t=self.inlet_temperature)
        s0 = get_s_from_state(p=self.inlet_pressure, h=h0)
        p1, v1t, t1t, h1t = F.compute_intermedia_point(h0, s0, stator_heat_drop)

        c1t = (2 * stator_heat_drop) ** 0.5
        a = get_sound_speed_from_state(h1t, s0)
        mach_1t = c1t / a
        if (mach_1t > 1).any():
            raise RuntimeError("M1t > 1")

        l1 = F.compute_stator_blade_length(
            self.mass_flow_rate, self.average_diameter, c1t, v1t, self.stator_outlet_angle, chord
        )
        fi = F.compute_speed_coefficient(blade_length=l1, chord=chord, is_rotor=False)
        nu1 = F.compute_discharge_coefficient(blade_length=l1, chord=chord, is_rotor=False)
        self.triangles.alpha_1 = F.move_angle(self.stator_outlet_angle, fi, nu1)

        self.triangles.c1 = fi * c1t
        self.triangles.w1, self.triangles.beta_1 = F.compute_triangle(
            u=self.triangles.u, inlet_speed=self.triangles.c1, inlet_angle=self.triangles.alpha_1)
        stator_loss = (c1t ** 2 - self.triangles.c1 ** 2) / 2

        h1 = h1t + stator_loss
        s1 = get_s_from_state(p=p1, h=h1)

        l2 = l1 + self.overlapping
        psi = F.compute_speed_coefficient(blade_length=l2, chord=chord, is_rotor=True)
        nu2 = F.compute_discharge_coefficient(blade_length=l2, chord=chord, is_rotor=True)
        _, v2t, t2t, h2t = F.compute_intermedia_point(h1, s1, rotor_heat_drop)
        w2t = (2 * rotor_heat_drop + self.triangles.w1 ** 2) ** 0.5

        a = get_sound_speed_from_state(h2t, s1)
        mach_2t = w2t / a
        if (mach_2t > 1).any():
            raise RuntimeError("M2t > 1")

        self.triangles.w2 = psi * w2t
        beta_2_eff = F.compute_beta_2(
            mass_flow_rate=self.mass_flow_rate, d=self.average_diameter, w2t=w2t, v2t=v2t, l2=l2, nu=nu2
        )
        self.triangles.beta_2 = F.move_angle(beta_2_eff, psi, nu2)

        self.triangles.c2, self.triangles.alpha_2 = F.compute_triangle(
            u=self.triangles.u, inlet_speed=self.triangles.w2, inlet_angle=self.triangles.beta_2
        )

        output_speed_loss = (self.triangles.c2 ** 2) / 2

        work_pu = self.triangles.u * (
                self.triangles.c1 * np.cos(self.triangles.alpha_1) + self.triangles.c2 * np.cos(self.triangles.alpha_2)
        )
        available_energy = total_heat_drop - int(1 - self.is_last) * output_speed_loss
        blade_efficiency = work_pu / available_energy
        power = work_pu * self.mass_flow_rate

        return blade_efficiency, u_div_c, power
