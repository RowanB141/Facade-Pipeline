import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')

import os
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

SCENE_XML = "/home/jesssms/projects/facade_pipeline/output/sionna_scene.xml"

def setup_scene():
    scene = load_scene(SCENE_XML)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.frequency = 3.5e9
    return scene

solver = PathSolver()

def compute_paths(scene, max_depth):
    return solver(scene, max_depth=max_depth)

# Experiment 1 — Non-LoS path validation
print("\n=== Experiment 1: Non-LoS Path Validation ===")
scene = setup_scene()
tx = Transmitter("tx", position=[10.0, -20.0, 5.0])
rx = Receiver("rx",    position=[0.0,   20.0, 2.0])
scene.add(tx); scene.add(rx)
paths = compute_paths(scene, max_depth=4)
a_real, a_imag = paths.a
print(f"  Paths found: {a_real.shape[-1]}")
print(f"  Path coefficients shape: {a_real.shape}")

# Experiment 2 — TX height sweep
print("\n=== Experiment 2: TX Height Sweep ===")
for height in [1.5, 10.0, 20.0, 35.0]:
    scene = setup_scene()
    tx = Transmitter("tx", position=[10.0, -20.0, height])
    rx = Receiver("rx",    position=[0.0,   20.0,  2.0])
    scene.add(tx); scene.add(rx)
    paths = compute_paths(scene, max_depth=4)
    a_real, a_imag = paths.a
    power = np.sum(np.array(a_real)**2 + np.array(a_imag)**2)
    print(f"  TX height={height:5.1f}m  paths={a_real.shape[-1]:3d}  power={power:.6f}")

# Experiment 3 — Reflections on vs off
print("\n=== Experiment 3: Reflections On vs Off ===")
scene = setup_scene()
tx = Transmitter("tx", position=[10.0, -20.0, 5.0])
rx = Receiver("rx",    position=[0.0,   20.0, 2.0])
scene.add(tx); scene.add(rx)

paths_on  = compute_paths(scene, max_depth=4)
paths_off = compute_paths(scene, max_depth=0)
ar_on,  ai_on  = paths_on.a
ar_off, ai_off = paths_off.a
power_on  = np.sum(np.array(ar_on)**2  + np.array(ai_on)**2)
power_off = np.sum(np.array(ar_off)**2 + np.array(ai_off)**2)
print(f"  Reflections ON  — paths={ar_on.shape[-1]}  power={power_on:.6f}")
print(f"  Reflections OFF — paths={ar_off.shape[-1]}  power={power_off:.6f}")

# Experiment 4 — Reflection depth 0 to 5
print("\n=== Experiment 4: Reflection Depth Analysis ===")
scene = setup_scene()
tx = Transmitter("tx", position=[10.0, -20.0, 5.0])
rx = Receiver("rx",    position=[0.0,   20.0, 2.0])
scene.add(tx); scene.add(rx)
for depth in range(6):
    paths = compute_paths(scene, max_depth=depth)
    a_real, a_imag = paths.a
    power = np.sum(np.array(a_real)**2 + np.array(a_imag)**2)
    print(f"  max_depth={depth}  paths={a_real.shape[-1]:3d}  cumulative_power={power:.6f}")

print("\n=== All experiments complete ===")
