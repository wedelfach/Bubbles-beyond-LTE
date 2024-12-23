# Bubbles-beyond-LTE
This repository contains tools necessary to reproduce data in [2]. The code was designed to compute steady-state bubble wall velocities beyond the local thermal equilibrium (LTE) approximation. The code "wall_velocity_frict.py" is written in Python. It is a modification of the code published in [1] that was designed to compute bubble walls in the LTE limit.
The main functions "find_vw" and "detonation" from the original code were augmented with the (optional) argument rho that quantifies the deviation from thermal equilibrium.
We also replaced the default scipy solver used in [1] with a custom one: slower but more reliable. For details check [2].

The code was validated with the results of the full hydrodynamic simulations, available at [3].

[1] "Model-independent bubble wall velocities in local thermal equilibrium." Journal of Cosmology and Astroparticle Physics 2023.07 (2023): 002.

[2] "Steady-state bubbles beyond the local-thermal equilibrium.", ArXiv: 2411.16580

[3] Repository https://drive.google.com/drive/folders/1vgFCIHZoppfbQbwkzDc549Tabukt0Zo_
