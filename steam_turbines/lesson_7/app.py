import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from iapws.iapws97 import IAPWS97

from turbine_stage import TurbineStage


def define_variable(name: str, vmin: float, vmax: float, vstep: float):
    _name = name.split(",")[0]
    st.markdown(f'### {_name}')
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        is_slider = st.toggle(value=False, label=f"{_name} to slider")
    if is_slider:
        variable = st.slider(f"Select {name}", min_value=vmin, max_value=vmax, value=vmin, step=vstep, format="%f")
    else:
        with col1:
            variable_min = st.number_input(f'Min {name}', min_value=vmin, max_value=vmax, step=vstep, format="%f")
        with col2:
            variable_max = st.number_input(f'Max {name}', min_value=vmin, max_value=vmax, step=vstep, format="%f")
        with col3:
            n_splits = st.number_input(f'N Splits {_name}',  min_value=1, max_value=50)
        variable = np.linspace(variable_min, variable_max, n_splits)
    st.divider()
    return variable


st.title('Steam Turbines Exercese #7')
st.markdown('## Axial stage designing process')
st.divider()
st.markdown('### Rotation speed')
rotation_speed = st.number_input("Rotation speed, rps", min_value=1.,  max_value=100., step=1.)
st.divider()
st.markdown('### Blade overlapping')
overlapping = st.number_input("Blade overlapping, m", min_value=0.001,  max_value=0.1, step=0.001, format="%f")
st.divider()
mass_flow_rate = define_variable("Mass flow rate, kg/s", vmin=1.0, vmax=2000.0, vstep=1.0)
inlet_pressure = define_variable("Inlet pressure, MPa", vmin=0.001, vmax=25.0, vstep=0.001,) * 1e6

if not isinstance(inlet_pressure, np.ndarray):
    t_lim = IAPWS97(P=inlet_pressure/1e6, x=0).T
    p_lim = inlet_pressure / 1e6
else:
    t_lim = 373.15
    p_lim = 23.0

inlet_temperature = define_variable("Inlet temperature, K", vmin=t_lim, vmax=700 + 273.15, vstep=1.0)
outlet_pressure = define_variable("Outlet pressure, MPa", vmin=0.001, vmax=p_lim, vstep=0.001) * 1e6
average_diameter = define_variable("Average diameter, m", vmin=0.2, vmax=4.0, vstep=0.01)
degree_of_reaction = define_variable("Degree of reaction, pu", vmin=0., vmax=0.9, vstep=0.001)
stator_outlet_angle = np.deg2rad(define_variable("Stator effective angle, deg", vmin=7.0, vmax=50., vstep=0.1))

if isinstance(degree_of_reaction, np.ndarray):
    inlet_pressure = np.ones_like(degree_of_reaction) * inlet_pressure
if isinstance(stator_outlet_angle, np.ndarray):
    inlet_pressure = np.ones_like(stator_outlet_angle) * inlet_pressure
if isinstance(average_diameter, np.ndarray):
    inlet_pressure = np.ones_like(average_diameter) * inlet_pressure


stage = TurbineStage(
    mass_flow_rate=mass_flow_rate,
    average_diameter=average_diameter,
    inlet_pressure=inlet_pressure,
    inlet_temperature=inlet_temperature,
    outlet_pressure=outlet_pressure,
    inlet_speed=0,
    degree_of_reaction=degree_of_reaction,
    overlapping=overlapping,
    rotation_speed=rotation_speed,
    stator_outlet_angle=stator_outlet_angle,
    is_last=True
)

x_axis = st.selectbox(
    'X axis is:',
    ('U/Сф', 'Degree of reaction', 'Alpha 1', "Average Diameter")
)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
if x_axis == 'U/Сф':
    ax.plot(stage.u_div_c, stage.blade_efficiency * 100)
if x_axis == "Degree of reaction":
    ax.plot(stage.degree_of_reaction, stage.blade_efficiency * 100)
if x_axis == "Alpha 1":
    ax.plot(stage.triangles.alpha_1, stage.blade_efficiency * 100)
if x_axis == "Average Diameter":
    ax.plot(stage.average_diameter, stage.blade_efficiency * 100)

ax.set_title(f"Blade efficiency = f({x_axis})")
ax.set_xlabel(x_axis)
ax.set_ylabel("Blade efficiency, %")
ax.grid()
st.pyplot(fig=fig)
