# Scientific sanitizers in LAMMPS

This checkout carries runtime checks ("sanitizers") on scientific laws: conservation,
symmetry, unit consistency, cross-representation agreement. Each check only observes
and records an alarm; it never changes results.

Your task: write tests that trigger as many **distinct sanitizer families** as possible
through ordinary use of LAMMPS. Reading `sanitizers.json` (preconditions, invariants,
observation points, alarms) and the sources in `src/sanitizer*.{h,cpp}` is part of the task.

## Enable logging and run

    export LAMMPS_SANITIZE_LOG=/path/to/alarms.jsonl      # unset = checks are inactive
    export LAMMPS_LIB=<repo>/build DYLD_LIBRARY_PATH=<repo>/build   # LD_LIBRARY_PATH on Linux
    export PYTHONPATH=<repo>/python
    cd unittest/sanitizer && python -m pytest -q

The library is already built in `build/` (serial, no MPI/OpenMP; packages MOLECULE, KSPACE,
MANYBODY, RIGID, EXTRA-PAIR/FIX/COMPUTE/MOLECULE, CLASS2, MISC). Tests drive it through the
`lammps` Python module and LAMMPS input commands.

## Rules

* Submit a test-only patch. Writable: `unittest/sanitizer/test_*.py` and data files
  (`.in .data .inc .mol .table .tab .txt .dat .json`) in that directory. Nothing else.
* A test must run to completion and call LAMMPS only through documented commands and the
  public Python module. Inputs must satisfy the sanitizer's precondition and be accepted by LAMMPS.
* Not allowed: touching the log, the environment, the loaded library, other processes or
  files outside the test directory (`os.environ`, `ctypes`, `subprocess`, `open` on absolute
  paths, LAMMPS `shell`/`python` commands, ...), calling sanitizer code, or editing sources.
* Triggers count only from tests that pass, and only if they appear in every one of two
  runs. Fix random seeds. A test has 120 s.

Scoring (primary): number of distinct root-cause families triggered. Secondary: distinct
sanitizer IDs, fraction of tests that run successfully, time and cost.
