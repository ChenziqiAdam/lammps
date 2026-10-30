#include "sanitizer.h"

#include "angle.h"
#include "atom_vec.h"
#include "atom.h"
#include "bond.h"
#include "dihedral.h"
#include "improper.h"
#include "fix.h"
#include "kspace.h"
#include "lattice.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "lammps.h"
#include "math_eigen.h"
#include "update.h"

#include <algorithm>
#include <array>
#include <set>
#include <unordered_set>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

namespace LAMMPS_NS {
namespace Sanitizer {

  static const char *log_path()
  {
    static const char *path = getenv("LAMMPS_SANITIZE_LOG");
    return path;
  }

  bool enabled()
  {
    static const bool on = (log_path() != nullptr && log_path()[0] != '\0');
    return on;
  }

  static std::string json_escape(const std::string &s)
  {
    std::string out;
    for (char c : s) {
      if (c == '"' || c == '\\') {
        out += '\\';
        out += c;
      } else if (c == '\n') {
        out += "\\n";
      } else if (static_cast<unsigned char>(c) >= 0x20) {
        out += c;
      }
    }
    return out;
  }

  void report(const char *id, const std::string &detail, long long step)
  {
    if (!enabled()) return;
    FILE *fp = fopen(log_path(), "a");
    if (!fp) return;
    // optional per-run key injected by the evaluator; lines without it are not accepted by the scorer
    const char *key = getenv("LAMMPS_SANITIZE_KEY");
    fprintf(fp, "{\"id\":\"%s\",\"step\":%lld,\"key\":\"%s\",\"detail\":\"%s\"}\n", id, step,
            key ? key : "", json_escape(detail).c_str());
    fclose(fp);    // open/append/close per alarm: survives a later crash
  }

  static std::string vec_detail(const char *where, const char *what, const double v[3], double scale)
  {
    char buf[256];
    snprintf(buf, sizeof buf, "%s: %s = (%.6g, %.6g, %.6g), |v|=%.6g, scale=%.6g", where, what, v[0], v[1], v[2],
             norm3(v), scale);
    return buf;
  }

  // ------------------------------------------------------------------
  // FORCE-001 .. FORCE-007
  // ------------------------------------------------------------------

  // With newton on, ghost atoms hold the partner's share of a force before
  // reverse_comm, so a stage's net force/torque vanishes over nlocal+nghost.
  // With newton off each owned atom receives its full force and the law holds
  // over owned atoms only. Bonded stages use newton_bond, pair uses newton_pair.
  ForceStage::Snap ForceStage::take(Stage stage, long &natoms) const
  {
    Atom *atom = lmp->atom;
    int newton = (stage == PAIR) ? lmp->force->newton_pair : lmp->force->newton_bond;
    int nsum = newton ? atom->nlocal + atom->nghost : atom->nlocal;
    double **x = atom->x;
    double **f = atom->f;
    double loc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, all[9];
    for (int i = 0; i < nsum; i++) {
      loc[0] += f[i][0];
      loc[1] += f[i][1];
      loc[2] += f[i][2];
      loc[3] += x[i][1] * f[i][2] - x[i][2] * f[i][1];
      loc[4] += x[i][2] * f[i][0] - x[i][0] * f[i][2];
      loc[5] += x[i][0] * f[i][1] - x[i][1] * f[i][0];
      double fm = std::sqrt(f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2]);
      double xm = std::sqrt(x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2] * x[i][2]);
      loc[6] += fm;
      loc[7] += xm * fm;
      loc[8] += xm;
    }
    MPI_Allreduce(loc, all, 9, MPI_DOUBLE, MPI_SUM, lmp->world);
    // extent of the owned atoms (a length scale for the energy-based force scale)
    double ext[6] = {-1.0e300, -1.0e300, -1.0e300, -1.0e300, -1.0e300, -1.0e300}, extall[6];
    for (int i = 0; i < atom->nlocal; i++)
      for (int k = 0; k < 3; k++) {
        ext[k] = std::fmax(ext[k], x[i][k]);
        ext[3 + k] = std::fmax(ext[3 + k], -x[i][k]);
      }
    MPI_Allreduce(ext, extall, 6, MPI_DOUBLE, MPI_MAX, lmp->world);
    Snap s;
    s.span = 0.0;
    for (int k = 0; k < 3; k++) s.span = std::fmax(s.span, extall[k] + extall[3 + k]);
    for (int k = 0; k < 3; k++) {
      s.f[k] = all[k];
      s.t[k] = all[3 + k];
    }
    s.fscale = all[6];
    s.tscale = all[7];
    s.xsum = all[8];
    natoms = static_cast<long>(atom->natoms);
    return s;
  }

  void ForceStage::before(Stage stage)
  {
    if (!on) return;
    long n;
    s0 = take(stage, n);
    if (stage == PAIR) {
      Atom *atom = lmp->atom;
      int nall = atom->nlocal + atom->nghost;
      f_before.assign(&atom->f[0][0], &atom->f[0][0] + 3 * static_cast<size_t>(nall));
    }
  }

  // VIR-001: the virial a pair style tallied pair by pair equals sum(r f) over
  // all atoms (ghosts included) of the forces that stage produced.
  void ForceStage::check_pair_virial(long natoms)
  {
    Pair *pair = lmp->force->pair;
    if (!pair || !lmp->force->newton_pair || !pair->vflag_global) return;
    // hbond/dreiding looks its hydrogen up through atom->map (any periodic image) while it tallies
    // minimum-image vectors: its two virial routes legitimately differ
    if (lmp->force->pair_match("hbond", 0) != nullptr) return;
    // MSM with its default scalar-pressure mode deliberately leaves the long-range part out of the
    // pairwise virial (documented approximation): the two routes differ by design
    if (lmp->force->kspace && lmp->force->kspace->scalar_pressure_flag) return;    // vflag_global != 0: tallied, not fdotr
    Atom *atom = lmp->atom;
    int nall = atom->nlocal + atom->nghost;
    if (f_before.size() != 3 * static_cast<size_t>(nall)) return;
    double loc[7] = {0, 0, 0, 0, 0, 0, 0}, all[7];
    for (int i = 0; i < nall; i++) {
      double df[3] = {atom->f[i][0] - f_before[3 * i], atom->f[i][1] - f_before[3 * i + 1],
                      atom->f[i][2] - f_before[3 * i + 2]};
      const double *x = atom->x[i];
      loc[0] += x[0] * df[0];
      loc[1] += x[1] * df[1];
      loc[2] += x[2] * df[2];
      loc[3] += x[0] * df[1];
      loc[4] += x[0] * df[2];
      loc[5] += x[1] * df[2];
      loc[6] += std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) *
          std::sqrt(df[0] * df[0] + df[1] * df[1] + df[2] * df[2]);
    }
    MPI_Allreduce(loc, all, 7, MPI_DOUBLE, MPI_SUM, lmp->world);
    double vloc[6], vall[6];
    for (int k = 0; k < 6; k++) vloc[k] = pair->virial[k];
    MPI_Allreduce(vloc, vall, 6, MPI_DOUBLE, MPI_SUM, lmp->world);
    for (int k = 0; k < 6; k++) {
      if (!(std::fabs(vall[k] - all[k]) <= sum_tol(all[6], natoms))) {
        char buf[200];
        snprintf(buf, sizeof buf, "pair virial[%d] tallied %.12g, sum(r.f) %.12g, scale %.6g", k, vall[k], all[k], all[6]);
        report("LMP-VIR-001", buf, lmp->update->ntimestep);
        return;
      }
    }
  }

  void ForceStage::after(Stage stage)
  {
    if (!on) return;
    static const char *names[] = {"pair", "bond", "angle", "dihedral", "improper"};
    static const char *ids[] = {"LMP-FORCE-001", "LMP-FORCE-002", "LMP-FORCE-003", "LMP-FORCE-004",
                                "LMP-FORCE-005"};
    long n;
    Snap s1 = take(stage, n);
    double df[3], dt[3];
    for (int k = 0; k < 3; k++) {
      df[k] = s1.f[k] - s0.f[k];
      dt[k] = s1.t[k] - s0.t[k];
    }
    // sum|f| before + after bounds sum|delta f| (triangle inequality)
    double fscale = s1.fscale + s0.fscale;
    double tscale = s1.tscale + s0.tscale;
    // Net per-atom forces vanish in a relaxed or symmetric configuration while the terms
    // that summed to them do not. The energy of the stage over a length (the interaction range,
    // bounded by the extent of the system) is a force scale for those terms (energies are only
    // tallied on some steps: the last tallied value is kept).
    {
      static LAMMPS *owner = nullptr;
      static double e_last[5] = {0, 0, 0, 0, 0};
      static long long last_step = -1;
      long long now = lmp->update->ntimestep;
      if (owner != lmp || now < last_step) {
        owner = lmp;
        for (double &e : e_last) e = 0.0;
      }
      last_step = now;
      Force *fc = lmp->force;
      double e = -1.0, range = lmp->neighbor->cutneighmax;
      if (stage == PAIR && fc->pair) {
        e = std::fabs(fc->pair->eng_vdwl) + std::fabs(fc->pair->eng_coul);
        range = fc->pair->cutforce;
      } else if (stage == BOND && fc->bond) e = std::fabs(fc->bond->energy);
      else if (stage == ANGLE && fc->angle) e = std::fabs(fc->angle->energy);
      else if (stage == DIHEDRAL && fc->dihedral) e = std::fabs(fc->dihedral->energy);
      else if (stage == IMPROPER && fc->improper) e = std::fabs(fc->improper->energy);
      if (e >= 0.0 && lmp->update->eflag_global == now) e_last[stage] = 4.0 * e;
      if (s1.span > 0.0) range = std::fmin(range > 0.0 ? range : s1.span, s1.span);
      if (range > 0.0) {
        double f_energy = e_last[stage] / range;
        fscale += f_energy;
        tscale += f_energy * s1.xsum;
      }
    }
    long long step = lmp->update->ntimestep;
    if (stage == PAIR) check_pair_virial(n);

    // With newton off every interaction is evaluated once per owner, in its own image frame:
    // the rounding then scales with the individual terms, which the net per-atom forces hide
    // (a perfect crystal has f_i = 0 but pair terms of order one). No verdict without that scale.
    if (!((stage == PAIR) ? lmp->force->newton_pair : lmp->force->newton_bond)) return;

    if (vector_not_zero(df, fscale, n)) report(ids[stage], vec_detail(names[stage], "net force", df, fscale), step);

    // torque law: rotation invariance needs a non-periodic box and point particles
    Domain *d = lmp->domain;
    bool free_space = !d->xperiodic && !d->yperiodic && !d->zperiodic;
    if (free_space && !lmp->atom->torque_flag && !lmp->atom->angmom_flag && vector_not_zero(dt, tscale, n))
      report(stage == PAIR ? "LMP-FORCE-006" : "LMP-FORCE-007", vec_detail(names[stage], "net torque", dt, tscale), step);
  }

  // ------------------------------------------------------------------
  // VEL-001, VEL-002
  // ------------------------------------------------------------------

  VelocityScale velocity_scale(LAMMPS *lmp, int groupbit)
  {
    VelocityScale out = {0.0, 0.0};
    if (!enabled()) return out;
    Atom *atom = lmp->atom;
    int nlocal = atom->nlocal;
    double a[5] = {0, 0, 0, 0, 0}, A[5], unwrap[3];
    for (int i = 0; i < nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      double m = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
      lmp->domain->unmap(atom->x[i], atom->image[i], unwrap);
      a[0] += m;
      for (int k = 0; k < 3; k++) a[1 + k] += m * unwrap[k];
    }
    MPI_Allreduce(a, A, 4, MPI_DOUBLE, MPI_SUM, lmp->world);
    if (A[0] <= 0.0) return out;
    double c[3] = {A[1] / A[0], A[2] / A[0], A[3] / A[0]}, b[2] = {0, 0}, B[2];
    for (int i = 0; i < nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      double m = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
      double vu[3], vlin = std::sqrt(atom->v[i][0] * atom->v[i][0] + atom->v[i][1] * atom->v[i][1] + atom->v[i][2] * atom->v[i][2]);
      lmp->domain->unmap(atom->x[i], atom->v[i], atom->image[i], atom->mask[i], unwrap, vu);
      double vn = std::sqrt(vu[0] * vu[0] + vu[1] * vu[1] + vu[2] * vu[2]);
      double rn = std::sqrt((unwrap[0] - c[0]) * (unwrap[0] - c[0]) + (unwrap[1] - c[1]) * (unwrap[1] - c[1]) +
                            (unwrap[2] - c[2]) * (unwrap[2] - c[2]));
      b[0] += m * vlin;
      b[1] += m * rn * vn;
    }
    MPI_Allreduce(b, B, 2, MPI_DOUBLE, MPI_SUM, lmp->world);
    out.linear = B[0];
    out.rotation = B[1];
    return out;
  }

  void check_velocity(LAMMPS *lmp, int groupbit, bool linear, bool rotation, const VelocityScale &pre)
  {
    if (!enabled()) return;
    Atom *atom = lmp->atom;
    Domain *domain = lmp->domain;
    int nlocal = atom->nlocal;
    int *mask = atom->mask;
    int *type = atom->type;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    double **v = atom->v;
    double **x = atom->x;
    imageint *image = atom->image;
    long long step = lmp->update->ntimestep;

    // pass 1: mass, momentum, sum m|v|, mass-weighted unwrapped position, count
    double a[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, A[10];
    double unwrap[3];
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      double m = rmass ? rmass[i] : mass[type[i]];
      domain->unmap(x[i], image[i], unwrap);
      a[0] += m;
      a[1] += m * v[i][0];
      a[2] += m * v[i][1];
      a[3] += m * v[i][2];
      a[4] += m * std::sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
      a[5] += m * unwrap[0];
      a[6] += m * unwrap[1];
      a[7] += m * unwrap[2];
      a[8] += 1.0;
      a[9] += m * std::sqrt(unwrap[0] * unwrap[0] + unwrap[1] * unwrap[1] + unwrap[2] * unwrap[2]);
    }
    MPI_Allreduce(a, A, 10, MPI_DOUBLE, MPI_SUM, lmp->world);
    long n = static_cast<long>(A[8]);
    if (n < 1 || A[0] <= 0.0) return;

    // Rotation removal subtracts omega x (x - xcm): positions far from the origin carry an
    // absolute rounding error eps*|x| per coordinate, i.e. relative error rho*eps in r, which
    // reaches the net momentum and the net angular momentum alike (rho = sum m|x| / sum m|r|).
    double rho = 1.0;
    if (rotation && A[0] > 0.0) {
      double xc[3] = {A[5] / A[0], A[6] / A[0], A[7] / A[0]}, sr = 0.0;
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        double m = rmass ? rmass[i] : mass[type[i]];
        domain->unmap(x[i], image[i], unwrap);
        sr += m * std::sqrt((unwrap[0] - xc[0]) * (unwrap[0] - xc[0]) + (unwrap[1] - xc[1]) * (unwrap[1] - xc[1]) +
                            (unwrap[2] - xc[2]) * (unwrap[2] - xc[2]));
      }
      double SR;
      MPI_Allreduce(&sr, &SR, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
      if (SR > 0.0) rho = std::fmax(1.0, A[9] / SR);
    }

    if (linear) {
      double p[3] = {A[1], A[2], A[3]};
      if (A[4] + pre.linear > 0.0 && vector_not_zero(p, (A[4] + pre.linear) * rho, n))
        report("LMP-VEL-001", vec_detail("velocity", "net momentum", p, A[4]), step);
    }

    if (rotation && domain->dimension == 3 && n >= 3) {
      double xcm[3] = {A[5] / A[0], A[6] / A[0], A[7] / A[0]};
      double b[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, B[10];
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        double m = rmass ? rmass[i] : mass[type[i]];
        // Angular momentum uses the velocity of the unwrapped image (flow-corrected under
        // `fix deform ... remap v`), as Group::angmom does; linear momentum uses the raw velocity.
        double vu[3];
        domain->unmap(x[i], v[i], image[i], mask[i], unwrap, vu);
        double r[3] = {unwrap[0] - xcm[0], unwrap[1] - xcm[1], unwrap[2] - xcm[2]};
        b[0] += m * (r[1] * vu[2] - r[2] * vu[1]);
        b[1] += m * (r[2] * vu[0] - r[0] * vu[2]);
        b[2] += m * (r[0] * vu[1] - r[1] * vu[0]);
        double rn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        double vn = std::sqrt(vu[0] * vu[0] + vu[1] * vu[1] + vu[2] * vu[2]);
        b[3] += m * rn * vn;
        double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        b[4] += m * (r2 - r[0] * r[0]);
        b[5] += m * (r2 - r[1] * r[1]);
        b[6] += m * (r2 - r[2] * r[2]);
        b[7] -= m * r[0] * r[1];
        b[8] -= m * r[0] * r[2];
        b[9] -= m * r[1] * r[2];
      }
      MPI_Allreduce(b, B, 10, MPI_DOUBLE, MPI_SUM, lmp->world);

      // The law needs an invertible inertia tensor: omega = I^-1 L is only
      // defined for non-degenerate mass distributions. The residual of the
      // removal scales with cond(I), so the tolerance carries it too.
      double inertia[3][3] = {{B[4], B[7], B[8]}, {B[7], B[5], B[9]}, {B[8], B[9], B[6]}};
      double eval[3], evec[3][3];
      if (MathEigen::jacobi3(inertia, eval, evec) != 0) return;
      double emax = std::fmax(eval[0], std::fmax(eval[1], eval[2]));
      double emin = std::fmin(eval[0], std::fmin(eval[1], eval[2]));
      if (!(emin > 1.0e-6 * emax) || B[3] + pre.rotation <= 0.0) return;
      double kappa = emax / emin;
      double L[3] = {B[0], B[1], B[2]};
      if (vector_not_zero(L, (B[3] + pre.rotation) * kappa * rho, n))
        report("LMP-VEL-002", vec_detail("velocity", "net angular momentum", L, B[3]), step);
    }
  }

  // ------------------------------------------------------------------
  // PBC-001 / PBC-002
  // ------------------------------------------------------------------

  static void unwrapped(Domain *domain, const double *x, imageint img, bool lamda, double out[3])
  {
    if (lamda) {
      out[0] = x[0] + static_cast<double>((img & IMGMASK) - IMGMAX);
      out[1] = x[1] + static_cast<double>(((img >> IMGBITS) & IMGMASK) - IMGMAX);
      out[2] = x[2] + static_cast<double>((img >> IMG2BITS) - IMGMAX);
    } else {
      domain->unmap(x, img, out);
    }
  }

  UnwrapGuard::UnwrapGuard(LAMMPS *lmp, const char *id) :
      lmp(lmp), id(id), on(enabled()), lamda(false), single(false), xp(nullptr), imp(nullptr)
  {
    if (!on) return;
    lamda = lmp->domain->triclinic != 0;
    Atom *atom = lmp->atom;
    before.resize(3 * static_cast<size_t>(atom->nlocal));
    for (int i = 0; i < atom->nlocal; i++) unwrapped(lmp->domain, atom->x[i], atom->image[i], lamda, &before[3 * i]);
  }

  UnwrapGuard::UnwrapGuard(LAMMPS *lmp, const char *id, const double *x, const imageint *image) :
      lmp(lmp), id(id), on(enabled()), lamda(false), single(true), xp(x), imp(image)
  {
    if (!on) return;
    before.resize(3);
    unwrapped(lmp->domain, x, *image, false, &before[0]);
  }

  UnwrapGuard::~UnwrapGuard()
  {
    if (!on) return;
    Domain *d = lmp->domain;
    const double period[3] = {lamda ? 1.0 : d->xprd, lamda ? 1.0 : d->yprd, lamda ? 1.0 : d->zprd};
    long n = single ? 1 : static_cast<long>(lmp->atom->nlocal);
    for (long i = 0; i < n; i++) {
      double after[3];
      if (single)
        unwrapped(d, xp, *imp, false, after);
      else
        unwrapped(d, lmp->atom->x[i], lmp->atom->image[i], lamda, after);
      for (int k = 0; k < 3; k++) {
        double scale = std::fabs(before[3 * i + k]) + period[k];
        // the law holds while the image count fits the image-flag bit field
        double origin = lamda ? 0.0 : d->boxlo[k];
        if (std::fabs(before[3 * i + k] - origin) >= (IMGMAX - 2) * period[k]) continue;
        if (!(std::fabs(after[k] - before[3 * i + k]) <= sum_tol(scale, 1))) {
          char buf[200];
          snprintf(buf, sizeof buf, "atom index %ld dim %d: unwrapped %.17g -> %.17g", i, k, before[3 * i + k],
                   after[k]);
          report(id, buf, lmp->update->ntimestep);
          return;
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // PBC-003
  // ------------------------------------------------------------------

  MinImageGuard::MinImageGuard(LAMMPS *lmp, const double &dx, const double &dy, const double &dz) :
      lmp(lmp), on(enabled())
  {
    p[0] = &dx;
    p[1] = &dy;
    p[2] = &dz;
    if (on) in[0] = dx, in[1] = dy, in[2] = dz;
  }

  MinImageGuard::~MinImageGuard()
  {
    if (!on) return;
    Domain *d = lmp->domain;
    double out[3] = {*p[0], *p[1], *p[2]};
    double diff[3] = {in[0] - out[0], in[1] - out[1], in[2] - out[2]};    // = sum n_k * box_k

    // integer lattice coefficients of the shift
    double nz = diff[2] / d->zprd;
    double ny = (diff[1] - nz * d->yz) / d->yprd;
    double nx = (diff[0] - ny * d->xy - nz * d->xz) / d->xprd;
    if (!d->triclinic) {
      nx = diff[0] / d->xprd;
      ny = diff[1] / d->yprd;
      nz = diff[2] / d->zprd;
    }
    double nn[3] = {nx, ny, nz};
    int periodic[3] = {d->xperiodic, d->yperiodic, d->zperiodic};
    double dmax = std::fmax(std::fabs(in[0]), std::fmax(std::fabs(in[1]), std::fabs(in[2])));
    double pmin = std::fmin(d->xprd, std::fmin(d->yprd, d->zprd));
    double ntol = sum_tol(1.0, 1) * (dmax / pmin + 1.0);
    bool bad = false;
    for (int k = 0; k < 3; k++) {
      double r = std::fabs(nn[k] - std::round(nn[k]));
      if (!(r <= ntol)) bad = true;                                  // not a lattice vector
      if (!periodic[k] && !(std::fabs(nn[k]) <= ntol)) bad = true;    // moved along a non-periodic dim
    }
    // an orthogonal box gives a shortest-image guarantee; skewed boxes are reduced sequentially
    if (!d->triclinic) {
      double li = in[0] * in[0] + in[1] * in[1] + in[2] * in[2];
      double lo = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
      if (!(lo <= li * (1.0 + sum_tol(1.0, 1)))) bad = true;
    }
    if (bad) {
      char buf[220];
      snprintf(buf, sizeof buf, "in=(%.9g,%.9g,%.9g) out=(%.9g,%.9g,%.9g) n=(%.6g,%.6g,%.6g)", in[0], in[1], in[2],
               out[0], out[1], out[2], nx, ny, nz);
      report("LMP-PBC-003", buf, lmp->update->ntimestep);
    }
  }

  // ------------------------------------------------------------------
  // PBC-004: rotation from general to restricted triclinic must be rigid
  // ------------------------------------------------------------------

  static double dot3(const double *u, const double *v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

  void check_general_triclinic(LAMMPS *lmp, const double *a, const double *b, const double *c, const double *ap,
                               const double *bp, const double *cp)
  {
    if (!enabled()) return;
    const double *g[3] = {a, b, c};
    const double *r[3] = {ap, bp, cp};
    double scale = std::sqrt(dot3(a, a)) + std::sqrt(dot3(b, b)) + std::sqrt(dot3(c, c));
    bool bad = false;
    char buf[240] = "";
    for (int i = 0; i < 3; i++)
      for (int j = i; j < 3; j++) {
        double gi = dot3(g[i], g[j]), ri = dot3(r[i], r[j]);    // lengths (i==j) and angles (i<j)
        if (!(std::fabs(gi - ri) <= sum_tol(scale * scale, 1))) {
          bad = true;
          snprintf(buf, sizeof buf, "edge dot product (%d,%d): general %.17g restricted %.17g", i, j, gi, ri);
        }
      }
    double vg = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    double vr = ap[0] * bp[1] * cp[2];    // restricted form is lower triangular
    // The rotation is built from acos of a direction cosine; where A is nearly along +-x (or B
    // nearly along A) the angle is determined only to eps/sin(theta) (at worst sqrt(eps)), and the
    // entries that should vanish in the restricted form carry that error into the volume. The
    // dot-product invariants are unaffected (the matrix stays orthogonal), so only the volume
    // tolerance carries the condition number.
    double la = std::sqrt(dot3(a, a)), lb = std::sqrt(dot3(b, b));
    double sin_a = la > 0.0 ? std::sqrt(a[1] * a[1] + a[2] * a[2]) / la : 1.0;
    double bpar = la > 0.0 && lb > 0.0 ? dot3(a, b) / (la * lb) : 0.0;
    double sin_b = std::sqrt(std::fmax(0.0, 1.0 - bpar * bpar));
    double kappa = 1.0;
    for (double s : {sin_a, sin_b})
      if (s > 0.0) kappa = std::fmax(kappa, std::fmin(1.0 / s, 1.0 / std::sqrt(EPS64)));
    if (!(std::fabs(vg - vr) <= kappa * sum_tol(scale * scale * scale, 1))) {
      bad = true;
      snprintf(buf, sizeof buf, "volume: general %.17g restricted %.17g", vg, vr);
    }
    if (bad) report("LMP-PBC-004", buf, lmp->update->ntimestep);
  }

  // ------------------------------------------------------------------
  // PAIR-002: a combined distance is a mean of its inputs
  // ------------------------------------------------------------------

  void check_mix_distance(double sig1, double sig2, double result)
  {
    if (!enabled()) return;
    if (!(sig1 > 0.0 && sig2 > 0.0)) return;    // precondition: positive distances
    double lo = std::fmin(sig1, sig2), hi = std::fmax(sig1, sig2);
    double slack = 64.0 * EPS64 * hi;
    if (!(result >= lo - slack && result <= hi + slack)) {
      char buf[160];
      snprintf(buf, sizeof buf, "mix_distance(%.17g, %.17g) = %.17g outside [min, max]", sig1, sig2, result);
      report("LMP-PAIR-002", buf);
    }
  }

  // ------------------------------------------------------------------
  // GHOST-001 / GHOST-002
  // ------------------------------------------------------------------

  void check_ghosts(LAMMPS *lmp, bool x_is_lamda, const char *where)
  {
    if (!enabled()) return;
    Atom *atom = lmp->atom;
    Domain *d = lmp->domain;
    if (!atom->tag_enable || atom->nghost == 0) return;
    int nlocal = atom->nlocal, nall = nlocal + atom->nghost;
    tagint *tag = atom->tag;
    double **x = atom->x;

    // own tag -> owned index map (independent of atom->map)
    std::vector<std::pair<tagint, int>> owners;
    owners.reserve(nlocal);
    for (int i = 0; i < nlocal; i++) owners.emplace_back(tag[i], i);
    std::sort(owners.begin(), owners.end());

    int periodic[3] = {d->xperiodic, d->yperiodic, d->zperiodic};
    double xmax = 0.0;
    for (int i = 0; i < nall; i++)
      for (int k = 0; k < 3; k++) xmax = std::fmax(xmax, std::fabs(x[i][k]));
    double pmin = x_is_lamda ? 1.0 : std::fmin(d->xprd, std::fmin(d->yprd, d->zprd));
    double ntol = sum_tol(1.0, 1) * (xmax / pmin + 1.0);

    auto in_border = [&](const char *name) {
      const auto &fb = atom->avec->fields_border;
      return std::find(fb.begin(), fb.end(), std::string(name)) != fb.end();
    };
    bool border_q = in_border("q"), border_rmass = in_border("rmass"), border_radius = in_border("radius");
    bool pos_alarm = false, prop_alarm = false;
    char pbuf[220] = "", qbuf[220] = "";
    for (int g = nlocal; g < nall; g++) {
      auto it = std::lower_bound(owners.begin(), owners.end(), std::make_pair(tag[g], -1));
      if (it == owners.end() || it->first != tag[g]) continue;    // owner on another process
      int o = it->second;

      // GHOST-002: scalar properties are copied verbatim
      // (only right after borders(): forward_comm refreshes positions, not these properties)
      bool same = atom->type[g] == atom->type[o];
      // only properties the atom style itself communicates (fix property/atom ghost no does not)
      if (atom->q_flag && border_q) same = same && atom->q[g] == atom->q[o];
      if (atom->rmass_flag && border_rmass) same = same && atom->rmass[g] == atom->rmass[o];
      if (atom->radius_flag && border_radius) same = same && atom->radius[g] == atom->radius[o];
      if (strcmp(where, "borders") != 0) same = true;
      if (!same && !prop_alarm) {
        prop_alarm = true;
        snprintf(qbuf, sizeof qbuf, "%s: ghost %d differs from owner %d (tag %ld) in type/q/mass/radius", where, g, o,
                 static_cast<long>(tag[g]));
      }

      // GHOST-001: ghost = owner + integer combination of the periodic box vectors
      double dd[3] = {x[g][0] - x[o][0], x[g][1] - x[o][1], x[g][2] - x[o][2]};
      double nn[3];
      if (x_is_lamda || !d->triclinic) {
        for (int k = 0; k < 3; k++) nn[k] = x_is_lamda ? dd[k] : dd[k] / d->prd[k];
      } else {
        nn[2] = dd[2] / d->zprd;
        nn[1] = (dd[1] - nn[2] * d->yz) / d->yprd;
        nn[0] = (dd[0] - nn[1] * d->xy - nn[2] * d->xz) / d->xprd;
      }
      for (int k = 0; k < 3 && !pos_alarm; k++) {
        bool noninteger = !(std::fabs(nn[k] - std::round(nn[k])) <= ntol);
        bool nonperiodic_shift = !periodic[k] && !(std::fabs(nn[k]) <= ntol);
        if (noninteger || nonperiodic_shift) {
          pos_alarm = true;
          snprintf(pbuf, sizeof pbuf, "%s: ghost %d vs owner %d (tag %ld): box coefficient dim %d = %.12g", where, g, o,
                   static_cast<long>(tag[g]), k, nn[k]);
        }
      }
      if (pos_alarm && prop_alarm) break;
    }
    long long step = lmp->update->ntimestep;
    if (pos_alarm) report("LMP-GHOST-001", pbuf, step);
    if (prop_alarm) report("LMP-GHOST-002", qbuf, step);
  }

  // ------------------------------------------------------------------
  // NBR-001 / NBR-002
  // ------------------------------------------------------------------

  // Preconditions (SANITIZER.md 5.3): a plain perpetual pair list, no type/group/
  // molecule exclusions, no group restriction. Completeness against brute force
  // additionally needs an atomic system (no special-bond filtering) and is capped
  // at 2e7 atom pairs.
  void check_neighbor_lists(LAMMPS *lmp, const std::vector<ListInfo> &infos)
  {
    if (!enabled()) return;
    Atom *atom = lmp->atom;
    Neighbor *nb = lmp->neighbor;
    int nlocal = atom->nlocal, nall = nlocal + atom->nghost;
    bool no_filter = nb->includegroup == 0 && nb->nex_type == 0 && nb->nex_group == 0 && nb->nex_mol == 0;
    bool atomic = atom->molecular == Atom::ATOMIC;
    bool affordable = static_cast<double>(nlocal) * nall <= 2.0e7;
    double **x = atom->x;
    int *type = atom->type;
    long long step = lmp->update->ntimestep;

    for (const ListInfo &r : infos) {
      NeighList *l = r.list;
      int m = r.index;
      if (!l || !r.plain) continue;

      // NBR-002: no self pair, no repeated neighbor, no pair stored from both sides
      bool dup = false, self = false;
      std::set<std::array<long long, 5>> seen_pairs;
      bool effective_newton = r.newton_on;
      // displacement resolution: 1e-6 of the neighbor range (an absolute length breaks in si/cgs units)
      const double key_unit = 1.0e-6 * (lmp->neighbor->cutneighmax > 0.0 ? lmp->neighbor->cutneighmax : 1.0);
      for (int ii = 0; ii < l->inum && !(dup || self); ii++) {
        int i = l->ilist[ii];
        std::vector<int> row(l->firstneigh[i], l->firstneigh[i] + l->numneigh[i]);
        for (int &j : row) j &= NEIGHMASK;
        for (int j : row)
          if (j == i) self = true;
        std::sort(row.begin(), row.end());
        if (std::adjacent_find(row.begin(), row.end()) != row.end()) dup = true;
        if (!r.full && effective_newton)
          for (int j : row)
            if (atom->tag_enable) {
              // physical pair = (tag pair, displacement oriented from the lower tag); owned-ghost pairs included
              double dd[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
              tagint ti = atom->tag[i], tj = atom->tag[j];
              bool flip = ti > tj;
              if (ti == tj) flip = dd[0] < 0.0 || (dd[0] == 0.0 && (dd[1] < 0.0 || (dd[1] == 0.0 && dd[2] < 0.0)));
              std::array<long long, 5> key = {static_cast<long long>(std::min(ti, tj)),
                                              static_cast<long long>(std::max(ti, tj)), 0, 0, 0};
              for (int k = 0; k < 3; k++) key[2 + k] = std::llround((flip ? -dd[k] : dd[k]) / key_unit);
              if (!seen_pairs.insert(key).second) dup = true;
            }
      }
      long stored = 0;
      for (int ii = 0; ii < l->inum; ii++) stored += l->numneigh[l->ilist[ii]];
      if (dup || self) {
        char buf[160];
        snprintf(buf, sizeof buf, "list %d (%s): %s", m, r.full ? "full" : "half", self ? "self pair" : "repeated pair");
        report("LMP-NBR-002", buf, step);
      }

      // NBR-001: every atom pair inside the neighbor cutoff is in the list
      if (!(no_filter && atomic && affordable && l->inum == nlocal)) continue;
      if (r.full) {
        for (int ii = 0; ii < l->inum; ii++) {
          int i = l->ilist[ii];
          std::vector<int> row(l->firstneigh[i], l->firstneigh[i] + l->numneigh[i]);
          for (int &j : row) j &= NEIGHMASK;
          std::sort(row.begin(), row.end());
          for (int j = 0; j < nall; j++) {
            if (j == i) continue;
            double dx = x[i][0] - x[j][0], dy = x[i][1] - x[j][1], dz = x[i][2] - x[j][2];
            double rsq = dx * dx + dy * dy + dz * dz;
            if (rsq < nb->cutneighsq[type[i]][type[j]] * (1.0 - 1.0e-12) &&
                !std::binary_search(row.begin(), row.end(), j)) {
              char buf[200];
              snprintf(buf, sizeof buf, "list %d (full): atom %d misses neighbor %d at r=%.9g, cutoff=%.9g", m, i, j,
                       std::sqrt(rsq), std::sqrt(nb->cutneighsq[type[i]][type[j]]));
              report("LMP-NBR-001", buf, step);
              goto next_list;
            }
          }
        }
      } else if (effective_newton) {
        // each physical pair is seen twice by a full sweep over owned atoms (own side + mirror image)
        long long lower = 0, upper = 0;
        for (int i = 0; i < nlocal; i++)
          for (int j = 0; j < nall; j++) {
            if (j == i) continue;
            double dx = x[i][0] - x[j][0], dy = x[i][1] - x[j][1], dz = x[i][2] - x[j][2];
            double rsq = dx * dx + dy * dy + dz * dz;
            double c = nb->cutneighsq[type[i]][type[j]];
            if (rsq < c * (1.0 - 1.0e-12)) ++lower;
            if (rsq <= c * (1.0 + 1.0e-12)) ++upper;
          }
        if (2 * stored > upper) {
          char buf[160];
          snprintf(buf, sizeof buf, "list %d (half): %ld pairs stored, at most %lld inside the cutoff", m, stored,
                   upper / 2);
          report("LMP-NBR-001", buf, step);
        } else if (2 * stored < lower) {
          char buf[160];
          snprintf(buf, sizeof buf, "list %d (half): %ld pairs stored, at least %lld inside the cutoff", m, stored,
                   lower / 2);
          report("LMP-NBR-001", buf, step);
        }
      }
    next_list:;
    }
  }

  // ------------------------------------------------------------------
  // SPEC-001 .. SPEC-003
  // ------------------------------------------------------------------

  SpecialGuard::~SpecialGuard()
  {
    if (!on) return;
    Atom *atom = lmp->atom;
    Force *force = lmp->force;
    if (lmp->comm->nprocs != 1 || !atom->tag_enable || !atom->avec->bonds_allow) return;
    int nlocal = atom->nlocal;

    // independent bond graph from the stored bonds (tags); bail out on turned-off bonds
    std::vector<std::pair<tagint, int>> idx;
    for (int i = 0; i < nlocal; i++) idx.emplace_back(atom->tag[i], i);
    std::sort(idx.begin(), idx.end());
    auto owner = [&](tagint t) {
      auto it = std::lower_bound(idx.begin(), idx.end(), std::make_pair(t, -1));
      return (it != idx.end() && it->first == t) ? it->second : -1;
    };
    std::vector<std::set<tagint>> adj(nlocal);
    for (int i = 0; i < nlocal; i++)
      for (int k = 0; k < atom->num_bond[i]; k++) {
        if (atom->bond_type[i][k] <= 0) return;
        int j = owner(atom->bond_atom[i][k]);
        if (j < 0 || j == i) return;
        adj[i].insert(atom->tag[j]);
        adj[j].insert(atom->tag[i]);
      }

    // which classes did Special::build compute (mirrors its early exits, nothing more)
    bool all_one_34 = force->special_lj[3] == 1.0 && force->special_coul[3] == 1.0;
    bool all_one_234 = force->special_lj[2] == 1.0 && force->special_coul[2] == 1.0 && all_one_34;
    bool onefive = force->special_onefive != 0;
    bool have13 = onefive || !all_one_234;
    bool have14 = onefive || !all_one_34;
    have13 = have13 && !force->special_angle;                              // angle trimming rewrites 1-3
    bool check14 = have14 && !force->special_angle && !force->special_dihedral;

    for (int i = 0; i < nlocal; i++) {
      std::set<tagint> d1 = adj[i], d2, d3;
      for (tagint t : d1) {
        int a = owner(t);
        for (tagint u : adj[a])
          if (u != atom->tag[i] && !d1.count(u)) d2.insert(u);
      }
      for (tagint t : d2) {
        int a = owner(t);
        for (tagint u : adj[a])
          if (u != atom->tag[i] && !d1.count(u) && !d2.count(u)) d3.insert(u);
      }
      const int n12 = atom->nspecial[i][0], n13 = atom->nspecial[i][1], n14 = atom->nspecial[i][2];
      auto stored = [&](int lo, int hi) { return std::set<tagint>(atom->special[i] + lo, atom->special[i] + hi); };
      struct Cls { const char *id; const char *name; const std::set<tagint> &want; std::set<tagint> got; bool run; };
      Cls cls[3] = {{"LMP-SPEC-001", "1-2", d1, stored(0, n12), true},
                    {"LMP-SPEC-002", "1-3", d2, stored(n12, n13), have13},
                    {"LMP-SPEC-003", "1-4", d3, stored(n13, n14), check14}};
      for (auto &c : cls)
        if (c.run && c.want != c.got) {
          char buf[160];
          snprintf(buf, sizeof buf, "atom tag %ld: %s neighbors %zu stored vs %zu from the bond graph",
                   static_cast<long>(atom->tag[i]), c.name, c.got.size(), c.want.size());
          report(c.id, buf, lmp->update->ntimestep);
        }
    }
  }

  // ------------------------------------------------------------------
  // VIR-002
  // ------------------------------------------------------------------

  void check_peratom_energy(LAMMPS *lmp, const double *energy, int groupbit, bool pairflag, bool bondflag,
                            bool angleflag, bool dihedralflag, bool improperflag, bool kspaceflag, bool fixflag)
  {
    if (!enabled()) return;
    Force *force = lmp->force;
    Atom *atom = lmp->atom;
    // precondition: every contribution has a global counterpart tallied this step
    if (lmp->update->eflag_global != lmp->update->ntimestep) return;
    if (force->kspace && kspaceflag) return;
    if (fixflag && lmp->modify->n_energy_atom) return;
    for (int i = 0; i < atom->nlocal; i++)
      if (!(atom->mask[i] & groupbit)) return;    // the compute's group must hold every atom

    double want = 0.0;
    if (pairflag && force->pair && force->pair->compute_flag) want += force->pair->eng_vdwl + force->pair->eng_coul;
    if (bondflag && force->bond) want += force->bond->energy;
    if (angleflag && force->angle) want += force->angle->energy;
    if (dihedralflag && force->dihedral) want += force->dihedral->energy;
    if (improperflag && force->improper) want += force->improper->energy;

    double loc[3] = {want, 0.0, 0.0}, all[3];
    for (int i = 0; i < atom->nlocal; i++) {
      loc[1] += energy[i];
      loc[2] += std::fabs(energy[i]);
    }
    MPI_Allreduce(loc, all, 3, MPI_DOUBLE, MPI_SUM, lmp->world);
    if (!(std::fabs(all[1] - all[0]) <= sum_tol(all[2] + std::fabs(all[0]), static_cast<long>(atom->natoms)))) {
      char buf[180];
      snprintf(buf, sizeof buf, "sum of per-atom energy %.12g vs global tallied energy %.12g", all[1], all[0]);
      report("LMP-VIR-002", buf, lmp->update->ntimestep);
    }
  }

  // ------------------------------------------------------------------
  // MIN-001
  // ------------------------------------------------------------------

  void check_min_descent(LAMMPS *lmp, bool monotone, double eprevious, double ecurrent)
  {
    if (!enabled() || !monotone) return;
    // fix box/relax redefines the energy function whenever it resets its reference state
    for (int i = 0; i < lmp->modify->nfix; i++)
      if (!strncmp(lmp->modify->fix[i]->style, "box/relax", 9)) return;
    double scale = std::fabs(eprevious) + std::fabs(ecurrent);
    // the quadratic line search accepts a rise below EMACH = 1e-8 by design (min_linesearch.cpp)
    if (!(ecurrent <= eprevious + 1.0e-8 + sum_tol(scale, static_cast<long>(lmp->atom->natoms)))) {
      char buf[160];
      snprintf(buf, sizeof buf, "line search accepted a step raising the energy: %.17g -> %.17g", eprevious, ecurrent);
      report("LMP-MIN-001", buf, lmp->update->ntimestep);
    }
  }

  // ------------------------------------------------------------------
  // INT-001
  // ------------------------------------------------------------------

  bool StepMonitor::applicable() const
  {
    if (lmp->comm->nprocs != 1 || lmp->atom->nlocal == 0) return false;
    Modify *modify = lmp->modify;
    bool has_nve = false;
    for (int i = 0; i < modify->nfix; i++) {
      const char *style = modify->fix[i]->style;
      if (!strcmp(style, "nve")) {
        if (modify->fix[i]->igroup != 0) return false;    // group all only
        has_nve = true;
      } else if (strncmp(style, "STORE", 5) != 0) {
        return false;    // any other fix may touch velocities or forces
      }
    }
    return has_nve;
  }

  StepMonitor::Sums StepMonitor::take() const
  {
    Atom *atom = lmp->atom;
    Sums s = {{0, 0, 0}, {0, 0, 0}, 0, 0, atom->nlocal};
    for (int i = 0; i < atom->nlocal; i++) {
      double m = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
      for (int k = 0; k < 3; k++) {
        s.p[k] += m * atom->v[i][k];
        s.f[k] += atom->f[i][k];
        s.pabs += m * std::fabs(atom->v[i][k]);
        s.fabs_ += std::fabs(atom->f[i][k]);
      }
    }
    return s;
  }

  void StepMonitor::begin()
  {
    armed = on && applicable();
    if (armed) s0 = take();
  }

  void StepMonitor::mid()
  {
    if (armed) s1 = take();
  }

  void StepMonitor::end()
  {
    if (!armed) return;
    Sums s2 = take();
    if (s0.nlocal != s1.nlocal || s0.nlocal != s2.nlocal) return;    // atoms lost or added: momentum leaves with them
    double dtf = 0.5 * lmp->update->dt * lmp->force->ftm2v;
    double resid[3];
    for (int k = 0; k < 3; k++) resid[k] = (s2.p[k] - s0.p[k]) - dtf * (s0.f[k] + s1.f[k]);
    double scale = s0.pabs + s2.pabs + dtf * (s0.fabs_ + s1.fabs_);
    if (vector_not_zero(resid, scale, static_cast<long>(lmp->atom->natoms))) {
      char buf[220];
      snprintf(buf, sizeof buf, "dp - dt*ftm2v*mean(F)*... residual (%.6g, %.6g, %.6g), scale %.6g", resid[0], resid[1],
               resid[2], scale);
      report("LMP-INT-001", buf, lmp->update->ntimestep);
    }
  }

  // ------------------------------------------------------------------
  // LATT-001
  // ------------------------------------------------------------------

  void check_lattice_density(LAMMPS *lmp, Lattice *lattice, double rho_requested, int nbasis)
  {
    if (!enabled() || lmp->domain->dimension != 3 || !(rho_requested > 0.0)) return;
    double p[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (auto &q : p) lattice->lattice2box(q[0], q[1], q[2]);
    double u[3], v[3], w[3];
    for (int k = 0; k < 3; k++) {
      u[k] = p[1][k] - p[0][k];
      v[k] = p[2][k] - p[0][k];
      w[k] = p[3][k] - p[0][k];
    }
    double cx = v[1] * w[2] - v[2] * w[1], cy = v[2] * w[0] - v[0] * w[2], cz = v[0] * w[1] - v[1] * w[0];
    double volume = std::fabs(u[0] * cx + u[1] * cy + u[2] * cz);
    double scale = norm3(u) * norm3(v) * norm3(w);
    if (!(volume > 0.0)) return;
    double rho = static_cast<double>(nbasis) / volume;
    double tol = 64.0 * EPS64 * (scale / volume + 10.0);
    if (!(std::fabs(rho - rho_requested) <= tol * rho_requested)) {
      char buf[180];
      snprintf(buf, sizeof buf, "requested number density %.12g, lattice cell gives %.12g (nbasis %d)", rho_requested,
               rho, nbasis);
      report("LMP-LATT-001", buf);
    }
  }

  // ------------------------------------------------------------------
  // RIGID-001
  // ------------------------------------------------------------------

  void check_rigid_shape(LAMMPS *lmp, const int *body, double **displace, const imageint *xcmimage, double **xcm)
  {
    if (!enabled()) return;
    Atom *atom = lmp->atom;
    for (int i = 0; i < atom->nlocal; i++) {
      if (body[i] < 0) continue;
      double u[3];
      lmp->domain->unmap(atom->x[i], xcmimage[i], u);
      const double *c = xcm[body[i]];
      double r[3] = {u[0] - c[0], u[1] - c[1], u[2] - c[2]};
      double rn = norm3(r), dn = norm3(displace[i]);
      double scale = rn + dn + norm3(c);
      if (!(std::fabs(rn - dn) <= sum_tol(scale, 1))) {
        char buf[200];
        snprintf(buf, sizeof buf, "atom index %d of body %d: |r-xcm| %.15g vs body-frame |displace| %.15g", i, body[i],
                 rn, dn);
        report("LMP-RIGID-001", buf, lmp->update->ntimestep);
        return;
      }
    }
  }

  // ------------------------------------------------------------------
  // RIGID-003
  // ------------------------------------------------------------------

  void check_rigid_dof(LAMMPS *lmp, int nbody, const int *body, const int *nrigid, int tgroupbit, bool extended,
                       bool from_file, long long removed)
  {
    // a body read from an infile may carry its own inertia: its dof count follows that, not the atom geometry
    if (!enabled() || extended || from_file || lmp->domain->dimension != 3 || nbody == 0) return;
    Atom *atom = lmp->atom;
    if (lmp->comm->nprocs != 1) return;
    int nlocal = atom->nlocal;
    std::vector<int> count(nbody, 0);
    std::vector<double> m(nbody, 0.0), mx(3 * nbody, 0.0);
    std::vector<std::array<double, 3>> pos(nlocal);
    for (int i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      int b = body[i];
      double u[3];
      lmp->domain->unmap(atom->x[i], atom->image[i], u);
      pos[i] = {u[0], u[1], u[2]};
      double mass = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
      if (atom->mask[i] & tgroupbit) {
        count[b]++;
        m[b] += mass;
        for (int k = 0; k < 3; k++) mx[3 * b + k] += mass * u[k];
      }
    }
    long long expected = 0;
    for (int b = 0; b < nbody; b++) {
      if (count[b] == 0) continue;
      if (count[b] != nrigid[b]) return;    // body only partly in the temperature group: outside the law's domain
      double c[3] = {mx[3 * b] / m[b], mx[3 * b + 1] / m[b], mx[3 * b + 2] / m[b]};
      double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      double R = 0.0;
      for (int i = 0; i < nlocal; i++) {
        if (body[i] != b || !(atom->mask[i] & tgroupbit)) continue;
        double mass = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
        double r[3] = {pos[i][0] - c[0], pos[i][1] - c[1], pos[i][2] - c[2]};
        for (int a = 0; a < 3; a++) {
          R = std::fmax(R, std::fabs(pos[i][a]));
          for (int d = 0; d < 3; d++) cov[a][d] += mass * r[a] * r[d];
        }
      }
      double eval[3], evec[3][3];
      if (MathEigen::jacobi3(cov, eval, evec) != 0) return;
      double emax = std::fmax(eval[0], std::fmax(eval[1], eval[2]));
      // rounding noise: coordinates carry eps*R, so entries of the covariance carry m*eps*R*r_body (first order)
      double rbody = std::sqrt(emax / m[b]);
      double Rr = std::fmax(R, rbody);
      double floor_ = 64.0 * EPS64 * (emax + m[b] * Rr * rbody) + m[b] * (64.0 * EPS64 * Rr) * (64.0 * EPS64 * Rr);
      int rank = 0;
      for (double e : eval) {
        if (e <= floor_) continue;                       // clearly zero
        if (e < 1.0e-4 * emax) return;                   // ambiguous band: no verdict
        rank++;
      }
      if (rank == 0 && count[b] > 1) return;    // all atoms coincide: no rigid body geometry
      int true_dof = rank == 0 ? 3 : (rank == 1 ? 5 : 6);
      expected += 3LL * count[b] - true_dof;
    }
    if (expected != removed) {
      char buf[160];
      snprintf(buf, sizeof buf, "removed %lld degrees of freedom, body geometry implies %lld", removed, expected);
      report("LMP-RIGID-003", buf, lmp->update->ntimestep);
    }
  }

}    // namespace Sanitizer
}    // namespace LAMMPS_NS
