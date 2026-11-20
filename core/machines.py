"""What a machine is (H49): its hardware and how it is run, and no physics. Each number here is a fact about the device, or a setting of its programme, with the source. The physics in coupled.py takes all of this as input.
FreeGSNKE's description of each machine's coils, vessel and wall lives in its own pickles (external/freegsnke/machine_configs, runs/twin/freeboundary/mast_machine2).
"""

import numpy as np


def miller(R0, a, kappa, delta, z0=0.0, n=24):
    """The boundary a machine's programme asks for, from its published shape numbers."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return (R0 + a * np.cos(th + np.arcsin(delta) * np.sin(th))).tolist(), (z0 + kappa * a * np.sin(th)).tolist()


_iter_R, _iter_Z = miller(6.2, 2.0, 1.85, 0.49, z0=0.6)      # 15 MA baseline: ITER Physics Basis, kappa_sep 1.85, delta_sep 0.49; X-point near (5.1, -3.25)
MACHINES = {
    "MAST": dict(
        R0=0.85, a=0.65, B0=0.52,
        tau_wall_s=0.005,               # the vessel's time for a low-n mode, order of magnitude
        density_feedback=False,         # a set gas programme in most of these shots: its density does not refill after a loss
        beams=dict(E_keV=65.0, R_tan=0.7, sigma=2.9e-20, f_lost="orbit"),        # SS and SW beams; the energy of the shots used
        free=dict(paths="runs/twin/freeboundary/mast_machine2", frozen=("P1", "P2", "P3", "P4", "P5"), feedback="BV", symmetric=True, R_out_flat=1.43),      # position kept by a far coil standing in for the machine's vertical-field feedback; 1.43 m is its usual outer edge in flat top, as a programme would state it
        # its pellet injector: cylinders, length equal to diameter, 1.1, 1.35 and 1.7 mm holding 0.6, 1.2 and 2.4e20 atoms, at 240 to 450 m/s, launched from the top
        # into the high field side (Valovic et al. 2008, J. Phys. Conf. Ser. 123 012039); 1.3 mm at about 300 m/s in 2011 (Valovic et al. 2013, arXiv:1305.2779,
        # whose "1.3e19 atoms" is ten times too few: a 1.3 mm cylinder of solid deuterium holds 1.0e20, and its figure 2 shows a 1.5e19 m^-3 rise over about 7 m^3)
        pellets=dict(d_mm=(1.1, 1.35, 1.7), default_d_mm=1.1, v_m_s=(240.0, 450.0), side="hfs"),
    ),
    "ITER": dict(
        R0=6.2, a=2.0, B0=5.3,
        tau_wall_s=0.2,
        free=dict(grid=dict(Rmin=3.2, Rmax=8.8, Zmin=-5.0, Zmax=5.0, nx=65, ny=65), z0=0.6, frozen=("CS1", "CS2", "CS3", "VS3"), limits=dict(CS1=24.9e6, CS2=24.9e6, CS3=24.9e6, VS3=6e4, PF1=11.9e6, PF2=6.3e6, PF3=10.2e6, PF4=9.3e6, PF5=11.3e6, PF6=23.9e6),
                  shape=dict(R=_iter_R, Z=_iter_Z, null_R=[5.07], null_Z=[-3.25])),      # limits in ampere-turns, from memory, to check against a published table
        # its pellet injection system: 5 mm fuelling pellets up to 10 Hz and 3 mm ELM-triggering pellets up to 20 Hz, from the inboard side (Baylor et al., ORNL,
        # the twin-screw extruder and pellet accelerator for ITER); "5 mm" read as a cylinder with length equal to diameter, as MAST's are given
        pellets=dict(d_mm=(3.0, 5.0), default_d_mm=5.0, side="hfs"),
    ),
    "MAST-U": dict(
        R0=0.85, a=0.53, B0=0.64,        # the upgrade's conventional double null, as FreeGSNKE's own example
        tau_wall_s=0.005,                # as MAST
        free=dict(grid=dict(Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129), z0=0.0, frozen=("Solenoid",), symmetric=True,
                  limits=dict(PX=5e3, D1=9e3, D2=9e3, D3=7e3, Dp=7e3, D5=5e3, D6=4e3, D7=5e3, P4=10e3, P5=10e3, P6=20e3, Solenoid=40e3),      # FreeGSNKE's example's limits, per turn; P6 is free there
                  shape=dict(R=[0.6, 0.6, 0.34, 1.4, 1.0, 1.0, 0.8, 0.8], Z=[1.1, -1.1, 0.0, 0.0, 2.0, -2.0, 1.62, -1.62], null_R=[0.6, 0.6], null_Z=[1.1, -1.1])),
    ),
    "SPARC": dict(
        R0=1.85, a=0.57, B0=12.2,        # primary reference discharge, Creely et al. 2020
        tau_wall_s=0.05,
        free=dict(grid=dict(Rmin=1.1, Rmax=2.7, Zmin=-1.8, Zmax=1.8, nx=65, ny=129), z0=0.0, frozen=("CS1I", "CS1O", "CS2", "CS3", "VS1"),       # the public description's coil turns are toy data (docs/lit/30), so its currents are not physical and its limits unknown
                  symmetric=True,
                  shape=dict(R=[1.542, 1.542, 2.42, 1.28, 1.7, 1.7], Z=[1.123, -1.123, 0.0, 0.0, 1.5, -1.5], null_R=[1.542, 1.542], null_Z=[1.123, -1.123])),      # the double null of FreeGSNKE's own SPARC example
    ),
}
