/* -*- c++ -*- ----------------------------------------------------------
   Scientific sanitizer runtime. Inactive unless the environment variable
   LAMMPS_SANITIZE_LOG names a file; then every alarm appends one JSON line.
   Sanitizers only observe: they never change results, raise errors or
   alter state.
------------------------------------------------------------------------- */

#ifndef LMP_SANITIZER_H
#define LMP_SANITIZER_H

#include "lmptype.h"
#include "sanitizer_checks.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {
class LAMMPS;
class NeighList;
class Lattice;
namespace Sanitizer {

  // true when LAMMPS_SANITIZE_LOG is set (evaluated once)
  bool enabled();

  // append one alarm record {"id","step","detail"}; no-op when disabled
  void report(const char *id, const std::string &detail, long long step = -1);

  // FORCE-001..007: one force stage (pair, bond, ...) judged in isolation.
  // before() and after() snapshot sum(f) and sum(x cross f) over the same
  // atom range; after() tests the difference.
  class ForceStage {
   public:
    enum Stage { PAIR, BOND, ANGLE, DIHEDRAL, IMPROPER };
    explicit ForceStage(LAMMPS *lmp) : lmp(lmp), on(enabled()) {}
    void before(Stage stage);
    void after(Stage stage);

   private:
    struct Snap {
      double f[3], t[3], fscale, tscale, xsum, span;
    };
    Snap take(Stage stage, long &natoms) const;
    void check_pair_virial(long natoms);
    LAMMPS *lmp;
    bool on;
    Snap s0 = {};
    std::vector<double> f_before;    // pair stage only: forces before the stage (VIR-001)
  };

  // VEL-001 / VEL-002 after `velocity create` / `velocity zero`.
  // Magnitude scales (sum m|v|, sum m|r||v|) of the velocities before an operation that
  // cancels momentum: rounding of the cancellation scales with the cancelled terms.
  struct VelocityScale {
    double linear, rotation;
  };
  VelocityScale velocity_scale(LAMMPS *lmp, int groupbit);
  void check_velocity(LAMMPS *lmp, int groupbit, bool linear, bool rotation, const VelocityScale &pre = {0.0, 0.0});

  // PBC-001 (all owned atoms, Domain::pbc) and PBC-002 (one atom, Domain::remap):
  // the unwrapped coordinate x + image*box must not change. `lamda` selects
  // fractional coordinates (Domain::pbc works in lamda space for triclinic boxes).
  class UnwrapGuard {
   public:
    UnwrapGuard(LAMMPS *lmp, const char *id);                  // all owned atoms
    UnwrapGuard(LAMMPS *lmp, const char *id, const double *x, const imageint *image);    // single atom
    ~UnwrapGuard();

   private:
    LAMMPS *lmp;
    const char *id;
    bool on, lamda, single;
    const double *xp;
    const imageint *imp;
    std::vector<double> before;
  };

  // PBC-003: minimum_image(dx,dy,dz) modifies its arguments in place.
  class MinImageGuard {
   public:
    MinImageGuard(LAMMPS *lmp, const double &dx, const double &dy, const double &dz);
    ~MinImageGuard();

   private:
    LAMMPS *lmp;
    bool on;
    const double *p[3];
    double in[3];
  };

  // PBC-004
  void check_general_triclinic(LAMMPS *lmp, const double *a, const double *b, const double *c, const double *ap,
                               const double *bp, const double *cp);

  // PAIR-002
  void check_mix_distance(double sig1, double sig2, double result);

  // GHOST-001 / GHOST-002; `x_is_lamda`: coordinates are fractional (triclinic, inside borders())
  void check_ghosts(LAMMPS *lmp, bool x_is_lamda, const char *where);

  // NBR-001 / NBR-002 after every perpetual pair list is built. The caller
  // (Neighbor, a friend of NeighRequest) fills this from the request flags.
  struct ListInfo {
    NeighList *list;
    int index;
    bool plain;        // perpetual atom-pair list without skip/cut/ghost/size/history/bond/rRESPA/copy/trim modifiers
    bool full;         // full list (else half)
    bool newton_on;    // effective newton setting of the list
  };
  void check_neighbor_lists(LAMMPS *lmp, const std::vector<ListInfo> &infos);

  // SPEC-001..003: constructed at the top of Special::build, checks on destruction
  class SpecialGuard {
   public:
    explicit SpecialGuard(LAMMPS *lmp) : lmp(lmp), on(enabled()) {}
    ~SpecialGuard();

   private:
    LAMMPS *lmp;
    bool on;
  };

  // VIR-002 at the end of ComputePEAtom::compute_peratom
  void check_peratom_energy(LAMMPS *lmp, const double *energy, int groupbit, bool pairflag, bool bondflag, bool angleflag,
                            bool dihedralflag, bool improperflag, bool kspaceflag, bool fixflag);

  // MIN-001: a successful line-search step must not raise the energy.
  // `monotone` is false for line searches without a sufficient-decrease rule.
  void check_min_descent(LAMMPS *lmp, bool monotone, double eprevious, double ecurrent);

  // INT-001: impulse-momentum theorem for one velocity-Verlet step,
  //   sum(m v_end) - sum(m v_start) = 0.5*dt*ftm2v * (sum f_start + sum f_end),
  // valid when `fix nve` on all atoms is the only fix acting on velocities.
  class StepMonitor {
   public:
    explicit StepMonitor(LAMMPS *lmp) : lmp(lmp), on(enabled()) {}
    void begin();    // before initial_integrate
    void mid();      // after the force computation, before final_integrate
    void end();      // after final_integrate
   private:
    struct Sums {
      double p[3], f[3], pabs, fabs_;
      int nlocal;
    };
    Sums take() const;
    bool applicable() const;
    LAMMPS *lmp;
    bool on, armed = false;
    Sums s0 = {}, s1 = {};
  };

  // LATT-001: in lj units the lattice scale is derived from a requested density;
  // the cell volume that lattice2box actually produces must give that density back.
  void check_lattice_density(LAMMPS *lmp, Lattice *lattice, double rho_requested, int nbasis);

  // RIGID-001: every atom of a rigid body stays at its body-frame distance from the center of mass.
  void check_rigid_shape(LAMMPS *lmp, const int *body, double **displace, const imageint *xcmimage, double **xcm);

  // RIGID-003: degrees of freedom removed by `fix rigid` match the body geometry
  // (3 for one atom, 5 for collinear atoms, 6 otherwise). `removed` is what FixRigid::dof returned.
  void check_rigid_dof(LAMMPS *lmp, int nbody, const int *body, const int *nrigid, int tgroupbit, bool extended,
                       bool from_file, long long removed);

}    // namespace Sanitizer
}    // namespace LAMMPS_NS

#endif
