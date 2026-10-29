/* -*- c++ -*- ----------------------------------------------------------
   Pure predicates for the scientific sanitizers. No LAMMPS state, no I/O,
   no logging: every function maps observed numbers to a verdict, so the same
   code can be exercised by a standalone sensitivity test.
------------------------------------------------------------------------- */

#ifndef LMP_SANITIZER_CHECKS_H
#define LMP_SANITIZER_CHECKS_H

#include <cmath>
#include <cstring>
#include <vector>

namespace LAMMPS_NS {
namespace Sanitizer {

  // true when |actual - expected| exceeds tol * |expected|; NaN counts as a mismatch
  inline bool rel_mismatch(double actual, double expected, double tol)
  {
    return !(std::fabs(actual - expected) <= tol * std::fabs(expected));
  }

  // ------------------------------------------------------------------
  // UNIT-001 .. UNIT-007: constants of `units` styles against accepted values
  // ------------------------------------------------------------------

  struct UnitConstants {
    double boltz, hplanck, mvv2e, ftm2v, nktv2p, qqr2e, angstrom, femtosecond, qelectron;
  };

  struct UnitViolation {
    const char *id;
    const char *what;
    double actual, expected;
  };

  // Tolerances. The constants in Update::set_units are stored with 5..10
  // significant digits and come from CODATA 2006; CODATA 2018 differs from
  // 2006 by at most 1.2e-6 relative for kB, h, e, N_A. UNIT_TOL is ~10x that
  // spread and above the rounding of the shortest stored literal (5 digits,
  // 5e-6). UNIT_PAIR_TOL covers the product of two independently typed
  // 9-digit literals (rounding <= 1e-8 each).
  constexpr double UNIT_TOL = 1.0e-5;
  constexpr double UNIT_PAIR_TOL = 1.0e-7;

  // CODATA 2018 and exact definitions
  constexpr double K_B = 1.380649e-23;             // J/K
  constexpr double H_PLANCK = 6.62607015e-34;      // J s
  constexpr double E_CHARGE = 1.602176634e-19;     // C
  constexpr double N_AVO = 6.02214076e23;          // 1/mol
  constexpr double COULOMB_K = 8.9875517923e9;     // 1/(4 pi eps0), N m^2 / C^2
  constexpr double BOHR = 5.29177210903e-11;       // m
  constexpr double HARTREE = 4.3597447222071e-18;  // J
  constexpr double AMU = 1.66053906660e-27;        // kg
  constexpr double STATC = 3.33564095198e-10;      // C per statcoulomb (esu)
  constexpr double KCAL = 4184.0;                  // J

  // Base units of each style expressed in SI. Returns false for `lj` or unknown.
  struct StyleUnits {
    double mass, length, time, energy, charge, pressure;
  };

  inline bool style_units(const char *style, StyleUnits &u)
  {
    if (!strcmp(style, "real")) {
      u = {1.0e-3 / N_AVO, 1.0e-10, 1.0e-15, KCAL / N_AVO, E_CHARGE, 101325.0};
    } else if (!strcmp(style, "metal")) {
      u = {1.0e-3 / N_AVO, 1.0e-10, 1.0e-12, E_CHARGE, E_CHARGE, 1.0e5};
    } else if (!strcmp(style, "si")) {
      u = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    } else if (!strcmp(style, "cgs")) {
      u = {1.0e-3, 1.0e-2, 1.0, 1.0e-7, STATC, 0.1};
    } else if (!strcmp(style, "electron")) {
      u = {AMU, BOHR, 1.0e-15, HARTREE, E_CHARGE, 1.0};
    } else if (!strcmp(style, "micro")) {
      u = {1.0e-15, 1.0e-6, 1.0e-6, 1.0e-15, 1.0e-12, 1.0e3};
    } else if (!strcmp(style, "nano")) {
      u = {1.0e-21, 1.0e-9, 1.0e-9, 1.0e-21, E_CHARGE, 1.0e6};
    } else
      return false;
    return true;
  }

  inline std::vector<UnitViolation> unit_violations(const char *style, const UnitConstants &c)
  {
    std::vector<UnitViolation> out;
    StyleUnits u;
    if (!style_units(style, u)) return out;

    // UNIT-001: energy->force->velocity conversions are inverse
    if (rel_mismatch(c.mvv2e * c.ftm2v, 1.0, UNIT_PAIR_TOL))
      out.push_back({"LMP-UNIT-001", "mvv2e*ftm2v", c.mvv2e * c.ftm2v, 1.0});

    // UNIT-002: Boltzmann constant in energy per kelvin
    double e_boltz = K_B / u.energy;
    if (rel_mismatch(c.boltz, e_boltz, UNIT_TOL)) out.push_back({"LMP-UNIT-002", "boltz", c.boltz, e_boltz});

    // UNIT-003: Planck constant in energy*time
    double e_h = H_PLANCK / (u.energy * u.time);
    if (rel_mismatch(c.hplanck, e_h, UNIT_TOL)) out.push_back({"LMP-UNIT-003", "hplanck", c.hplanck, e_h});

    // UNIT-004: Coulomb constant e^2/(4 pi eps0) in energy*length/charge^2
    double e_qqr2e = COULOMB_K * u.charge * u.charge / (u.length * u.energy);
    if (rel_mismatch(c.qqr2e, e_qqr2e, UNIT_TOL)) out.push_back({"LMP-UNIT-004", "qqr2e", c.qqr2e, e_qqr2e});

    // UNIT-005: (energy/volume) to pressure
    double e_nktv2p = u.energy / (u.length * u.length * u.length) / u.pressure;
    if (rel_mismatch(c.nktv2p, e_nktv2p, UNIT_TOL))
      out.push_back({"LMP-UNIT-005", "nktv2p", c.nktv2p, e_nktv2p});

    // UNIT-006: mass*(length/time)^2 to energy
    double e_mvv2e = u.mass * (u.length / u.time) * (u.length / u.time) / u.energy;
    if (rel_mismatch(c.mvv2e, e_mvv2e, UNIT_TOL)) out.push_back({"LMP-UNIT-006", "mvv2e", c.mvv2e, e_mvv2e});

    // UNIT-007: scale factors to SI length, time and elementary charge
    double e_ang = 1.0e-10 / u.length;
    double e_fs = 1.0e-15 / u.time;
    double e_qe = E_CHARGE / u.charge;
    if (rel_mismatch(c.angstrom, e_ang, UNIT_TOL)) out.push_back({"LMP-UNIT-007", "angstrom", c.angstrom, e_ang});
    if (rel_mismatch(c.femtosecond, e_fs, UNIT_TOL))
      out.push_back({"LMP-UNIT-007", "femtosecond", c.femtosecond, e_fs});
    if (rel_mismatch(c.qelectron, e_qe, UNIT_TOL))
      out.push_back({"LMP-UNIT-007", "qelectron", c.qelectron, e_qe});
    return out;
  }


  // ------------------------------------------------------------------
  // Summation tolerance (SANITIZER.md 5.8, T): a sum of n terms, each
  // carrying at most one rounding of its own size, differs from its exact
  // value by at most ~n*eps*sum|term|. The factor 64 absorbs accumulation
  // order and the 2-norm conversion; n is padded so tiny systems are not
  // held to the bare eps.
  // ------------------------------------------------------------------
  constexpr double EPS64 = 2.220446049250313e-16;

  inline double sum_tol(double scale, long n) { return 64.0 * EPS64 * (static_cast<double>(n) + 10.0) * scale; }

  inline double norm3(const double v[3]) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

  // FORCE-001..007, VEL-001, VEL-002: a 3-vector that the law says is zero,
  // against the sum of the absolute terms that built it. NaN alarms.
  inline bool vector_not_zero(const double v[3], double scale, long n)
  {
    return !(norm3(v) <= sum_tol(scale, n));
  }

}    // namespace Sanitizer
}    // namespace LAMMPS_NS

#endif
