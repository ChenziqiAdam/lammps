"""Template: a valid test. It exercises LAMMPS through the Python module only."""
from lammps import lammps


def test_lj_melt_runs():
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string("""
    units lj
    lattice fcc 0.8442
    region box block 0 3 0 3 0 3
    create_box 1 box
    create_atoms 1 box
    mass 1 1.0
    velocity all create 1.44 87287 loop geom
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0 2.5
    fix 1 all nve
    run 20
    """)
    assert lmp.get_natoms() == 108
    lmp.close()
