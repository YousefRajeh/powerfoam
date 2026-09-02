"""Matrix-free Gauss-Newton recovery of PowerFoam appearance coefficients.

Geometry (points / radii / quaternions / density) is frozen from a checkpoint; the unknowns are
`texel_sv_rgb`.  powerfoam/color_fn.py:95-98 computes

    value = (sum_i w_i * val_i) / (sum_i w_i) + 0.5 ,   then   max(value, 0)

with w_i depending only on axis/temp/direction, so the render is PIECEWISE AFFINE in the
coefficients -- affine wherever no texel crosses the ReLU.  That makes appearance fitting a
sparse linear least-squares problem on each active set, and Gauss-Newton applies.

Only the existing kernels are used:
    J v    = forward difference of the render along v, reusing the cached f(x)
    J^T u  = autograd.grad(render, coeffs, grad_outputs=u)     (the existing backward kernel)

Inner (preconditioned) CG on (J^T J + lam I) d = J^T r; outer Levenberg-Marquardt.

Three optimisations matter for wall clock, all controlled from the constructor:

  * `fx_cache`     -- f(x) is rendered once per GN step and reused, so the directional
                      derivative is a ONE-sided difference: 1 render per view per CG iteration
                      instead of 2.  Exact here, because f is affine in a neighbourhood of x
                      (the affinity probe measures how large that neighbourhood is).
  * `graph_reuse`  -- x is constant across the inner CG iterations, so in principle the
                      autograd graph per view could be built once per GN step and
                      re-differentiated, removing the forward render from J^T.  DISABLED BY
                      DEFAULT: powerfoam/rasterize.py's custom autograd Function cannot be
                      differentiated twice (`ctx.rasterizer` is gone on the second backward),
                      so enabling this needs a change there first.  Worth ~35% of CG time.
  * `precond`      -- Jacobi preconditioner on the normal equations, diag(J^T J) by Hutchinson
                      probes.  DISABLED BY DEFAULT: MEASURED HARMFUL.  With 9.9M unknowns a
                      handful of probes estimates the diagonal at roughly noise level, and the
                      resulting scaling is worse than none (8 GN steps reached 14.93 dB with it
                      vs 27.89 dB without, scene0062 truefrozen, 8 views).  Note also that
                      inner-solver conditioning is probably NOT why the per-step gains decay:
                      Ruhe, "Accelerated Gauss-Newton algorithms for nonlinear least squares
                      problems" (BIT 19, 1979) shows GN WITH A LINE SEARCH behaves
                      asymptotically like steepest descent on LARGE-RESIDUAL problems, with a
                      linear rate set by the conditioning of H = I - gamma*K.  The observed
                      per-step PSNR gains here decay geometrically at ratio ~0.75, which is
                      that behaviour.  Ruhe's remedy is conjugate-gradient acceleration of the
                      OUTER GN sequence, not a better inner preconditioner.

Used by gn_marimo.py (live view) and runnable as a CLI.
"""
import os
import sys
import time

import numpy as np
import torch
import configargparse

sys.path.insert(0, os.getcwd())


def _lazy_imports():
    import warp as wp

    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.metrics import psnr as psnr_fn

    return wp, Params, add_group, DataHandler, PowerfoamScene, psnr_fn


def parse_config(config_path, extra_argv=()):
    """Resolve a PowerFoam config file into an args namespace, as train.py does."""
    _, Params, add_group, _, _, _ = _lazy_imports()
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, required=True)
    argv = ["-c", str(config_path)] + list(extra_argv)
    a = parser.parse_args(argv)
    return get_params(a)


class GNSolver:
    """Holds the frozen scene, the view sets, and the LM/CG state."""

    def __init__(self, config_path, ckpt, n_views=8, n_held=12,
                 lam=1e-4, fd_eps=1e-2, seed=0,
                 graph_reuse=False, precond=False, precond_probes=4):
        wp, _, _, DataHandler, PowerfoamScene, psnr_fn = _lazy_imports()
        wp.init()
        torch.manual_seed(seed)
        self._psnr_fn = psnr_fn

        args = parse_config(config_path)
        self.args = args
        self.lam = lam
        self.fd_eps = fd_eps
        self.graph_reuse = graph_reuse
        self.use_precond = precond
        self.precond_probes = precond_probes

        split = "test" if args.eval else "all"
        dh = DataHandler(args)
        dh.reload(split, downsample=args.downsample[-1])
        self.dh = dh

        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device="cuda")
        model.declare_optimizers(args, args.iterations)
        model.sort_points()
        model.load_pt(ckpt)
        self.model = model

        n_avail = len(dh.cameras)
        idx = np.linspace(0, n_avail - 1, min(n_views, n_avail)).astype(int)
        self.fit_idx = idx.tolist()
        self.cams = [dh.cameras[i] for i in idx]
        self.gts = [dh.rgbs[i].cuda() for i in idx]

        rest = [i for i in range(n_avail) if i not in set(self.fit_idx)]
        if rest and n_held:
            step = max(1, len(rest) // n_held)
            self.held_idx = rest[::step][:n_held]
        else:
            self.held_idx = []
        self.hcams = [dh.cameras[i] for i in self.held_idx]
        self.hgts = [dh.rgbs[i].cuda() for i in self.held_idx]

        self.trained = model.texel_sv_rgb.detach().clone()
        self.shape = self.trained.shape
        self.dtype = self.trained.dtype
        self.n_unknowns = int(self.trained.numel())
        self.n_primitives = int(model.points.shape[0])
        self.n_views_avail = n_avail

        for p in model.parameters():
            p.requires_grad_(False)
        if "texel_sv_rgb" in model._parameters:
            del model._parameters["texel_sv_rgb"]

        self.x = torch.zeros(self.shape, device="cuda", dtype=self.dtype)
        self.total_cg = 0
        self.gn_step = 0
        self.history = []

        self._fx = None        # cached render at self.x
        self._graphs = None    # [(leaf, rgb)] retained graphs at self.x
        self._Minv = None      # Jacobi preconditioner, 1 / diag(J^T J)
        self.n_renders = 0     # forward renders, for honest cost accounting
        self.n_backwards = 0

        self.base_psnr = self.psnr(self.trained)
        self.base_held = self.psnr_held(self.trained)

    # ---- rendering helpers ------------------------------------------------------------
    def _render(self, coeffs, cams):
        self.model.texel_sv_rgb = coeffs
        with torch.no_grad():
            out = [self.model.forward(c)[0].detach() for c in cams]
        self.n_renders += len(cams)
        return out

    def render_fit(self, coeffs):
        return self._render(coeffs, self.cams)

    def fx(self):
        """f(x) at the current iterate, rendered once per GN step."""
        if self._fx is None:
            self._fx = self.render_fit(self.x)
        return self._fx

    def _invalidate(self):
        self._fx = None
        self._graphs = None

    def psnr_of(self, outs, gts):
        return float(np.mean([self._psnr_fn(o.clamp(0, 1), g).item()
                              for o, g in zip(outs, gts)]))

    def psnr(self, coeffs=None):
        outs = self.fx() if coeffs is None else self._render(coeffs, self.cams)
        return self.psnr_of(outs, self.gts)

    def psnr_held(self, coeffs=None):
        if not self.hcams:
            return float("nan")
        c = self.x if coeffs is None else coeffs
        return self.psnr_of(self._render(c, self.hcams), self.hgts)

    def sse_of(self, outs):
        return sum(float(((o - g) ** 2).sum()) for o, g in zip(outs, self.gts))

    def sse(self, coeffs=None):
        outs = self.fx() if coeffs is None else self._render(coeffs, self.cams)
        return self.sse_of(outs)

    def preview(self, view=0, coeffs=None):
        """(gt, gauss_newton, sgd) uint8 HxWx3 arrays for one fit view."""
        coeffs = self.x if coeffs is None else coeffs
        cam = [self.cams[view]]
        gn = self._render(coeffs, cam)[0].clamp(0, 1)
        sgd = self._render(self.trained, cam)[0].clamp(0, 1)
        gt = self.gts[view].clamp(0, 1)

        def to8(t):
            return (t.detach().cpu().numpy() * 255).astype(np.uint8)

        return to8(gt), to8(gn), to8(sgd)

    # ---- affinity probe ---------------------------------------------------------------
    def affinity(self, scales=(1.0, 1e-3)):
        """Relative second-difference of f along a random direction, per perturbation scale.

        Near zero => locally affine in the coefficients => the least-squares premise holds,
        and the one-sided difference used by Jv is exact rather than approximate.
        """
        zero = torch.zeros(self.shape, device="cuda", dtype=self.dtype)
        f0 = self.render_fit(zero)
        out = {}
        for s in scales:
            v = torch.randn(self.shape, device="cuda", dtype=self.dtype)
            v = v * self.trained.std() * s
            f1, f2 = self.render_fit(v), self.render_fit(2 * v)
            num = max(float((b - a - 2 * (c - a)).abs().max())
                      for a, c, b in zip(f0, f1, f2))
            den = max(float((c - a).abs().max()) for a, c in zip(f0, f1)) + 1e-12
            out[s] = num / den
        return out

    # ---- operators --------------------------------------------------------------------
    def Jv(self, vec):
        """One-sided directional derivative against the cached f(x): 1 render per view."""
        n = float(vec.norm()) + 1e-30
        h = self.fd_eps / n
        fp = self.render_fit(self.x + h * vec)
        return [(p - b) / h for p, b in zip(fp, self.fx())]

    def _build_graphs(self):
        """Retain one autograd graph per fit view at the current x."""
        graphs = []
        try:
            for cam in self.cams:
                leaf = self.x.clone().requires_grad_(True)
                self.model.texel_sv_rgb = leaf
                with torch.enable_grad():
                    rgb = self.model.forward(cam)[0]
                graphs.append((leaf, rgb))
            self.n_renders += len(self.cams)
            self._graphs = graphs
        except torch.cuda.OutOfMemoryError:
            del graphs
            torch.cuda.empty_cache()
            self.graph_reuse = False
            self._graphs = None

    def JTu(self, us):
        acc = torch.zeros(self.shape, device="cuda", dtype=self.dtype)
        if self.graph_reuse:
            if self._graphs is None:
                self._build_graphs()
            if self._graphs is not None:
                for (leaf, rgb), u in zip(self._graphs, us):
                    (g,) = torch.autograd.grad(rgb, leaf, grad_outputs=u,
                                               retain_graph=True)
                    acc += g
                self.n_backwards += len(self._graphs)
                return acc
        for cam, u in zip(self.cams, us):
            leaf = self.x.clone().requires_grad_(True)
            self.model.texel_sv_rgb = leaf
            with torch.enable_grad():
                rgb = self.model.forward(cam)[0]
                (g,) = torch.autograd.grad(rgb, leaf, grad_outputs=u)
            acc += g
            del leaf, rgb, g
        self.n_renders += len(self.cams)
        self.n_backwards += len(self.cams)
        return acc

    def _A(self, p):
        return self.JTu(self.Jv(p)) + self.lam * p

    def build_precond(self):
        """Jacobi preconditioner: diag(J^T J) by Hutchinson probes with Rademacher vectors."""
        acc = torch.zeros(self.shape, device="cuda", dtype=self.dtype)
        for _ in range(self.precond_probes):
            z = torch.randint(0, 2, self.shape, device="cuda",
                              dtype=self.dtype) * 2 - 1
            acc += z * self._A(z)
        d = (acc / self.precond_probes).clamp_min(0)
        pos = d[d > 0]
        floor = float(pos.median()) * 1e-3 if pos.numel() else 1.0
        self._Minv = 1.0 / d.clamp_min(max(floor, 1e-12))
        return self._Minv

    # ---- the loop ---------------------------------------------------------------------
    def steps(self, gn_iters=8, cg_iters=15, cg_tol=1e-3):
        """Generator yielding a dict after each inner CG iteration and each outer GN step."""
        t0 = time.time()
        yield dict(kind="gn", gn=0, cg=0, sse=self.sse(), psnr=self.psnr(),
                   psnr_held=self.psnr_held(), lam=self.lam, accepted=True,
                   elapsed=0.0, renders=self.n_renders, backwards=self.n_backwards)

        if self.use_precond and self._Minv is None:
            self.build_precond()

        for gn in range(1, gn_iters + 1):
            fx = self.fx()
            prev = self.sse_of(fx)
            resid = [g - o for o, g in zip(fx, self.gts)]
            rhs = self.JTu(resid)
            rhs_norm = float(rhs.norm())

            d = torch.zeros_like(self.x)
            r = rhs.clone()
            z = r * self._Minv if self._Minv is not None else r
            p = z.clone()
            rz = float((r * z).sum())

            for _ in range(cg_iters):
                Ap = self._A(p)
                pAp = float((p * Ap).sum())
                if pAp <= 0:
                    yield dict(kind="cg", gn=gn, cg=self.total_cg,
                               resid=float(r.norm()), rhs_norm=rhs_norm,
                               note="curvature breakdown", elapsed=time.time() - t0)
                    break
                al = rz / pAp
                d = d + al * p
                r = r - al * Ap
                z = r * self._Minv if self._Minv is not None else r
                rz_new = float((r * z).sum())
                p = z + (rz_new / rz) * p
                rz = rz_new
                self.total_cg += 1
                rn = float(r.norm())
                yield dict(kind="cg", gn=gn, cg=self.total_cg, resid=rn,
                           rhs_norm=rhs_norm, elapsed=time.time() - t0)
                if rn < cg_tol * rhs_norm:
                    break

            # Levenberg-Marquardt accept / halve
            new, accepted = prev, False
            for _ in range(6):
                trial = self.x + d
                cand_out = self.render_fit(trial)
                cand = self.sse_of(cand_out)
                if cand < prev:
                    self.x = trial
                    self._invalidate()
                    self._fx = cand_out
                    new, accepted = cand, True
                    self.lam = max(self.lam * 0.5, 1e-9)
                    break
                d = d * 0.5
                self.lam = self.lam * 4.0
            if not accepted:
                self._invalidate()

            self.gn_step = gn
            rec = dict(kind="gn", gn=gn, cg=self.total_cg, sse=new,
                       psnr=self.psnr(), psnr_held=self.psnr_held(),
                       lam=self.lam, accepted=accepted, elapsed=time.time() - t0,
                       renders=self.n_renders, backwards=self.n_backwards)
            self.history.append(rec)
            yield rec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--held", type=int, default=12)
    parser.add_argument("--gn_iters", type=int, default=8)
    parser.add_argument("--cg_iters", type=int, default=15)
    parser.add_argument("--lam", type=float, default=1e-4)
    parser.add_argument("--fd_eps", type=float, default=1e-2)
    parser.add_argument("--graph_reuse", action="store_true",
                        help="needs a double-backward-safe rasterize.py")
    parser.add_argument("--precond", action="store_true",
                        help="Jacobi/Hutchinson; measured harmful, see module docstring")
    a, _ = parser.parse_known_args(argv)

    s = GNSolver(a.config, a.ckpt, n_views=a.views, n_held=a.held,
                 lam=a.lam, fd_eps=a.fd_eps,
                 graph_reuse=a.graph_reuse, precond=a.precond)
    print("primitives {:,}   unknowns {:,}   fit views {}   held views {}".format(
        s.n_primitives, s.n_unknowns, len(s.cams), len(s.hcams)), flush=True)
    print("graph_reuse={}   precond={}".format(s.graph_reuse, s.use_precond), flush=True)
    for scale, val in s.affinity().items():
        verdict = "locally AFFINE" if val < 1e-2 else "ReLU clamps active"
        print("[affinity] scale={:<8g} nonlinearity={:.3e}  -> {}".format(
            scale, val, verdict), flush=True)
    print("[baseline] SGD coefficients: fit {:.3f} dB   held {:.3f} dB".format(
        s.base_psnr, s.base_held), flush=True)
    print("")
    print("{:>3}{:>6}{:>13}{:>9}{:>9}{:>10}{:>9}{:>8}".format(
        "GN", "CGtot", "sqrt(SSE)", "PSNR", "held", "elapsed_s", "renders", "bwd"))
    print("-" * 67, flush=True)
    for ev in s.steps(a.gn_iters, a.cg_iters):
        if ev["kind"] != "gn":
            continue
        print("{:>3}{:>6}{:>13.4e}{:>9.3f}{:>9.3f}{:>10.1f}{:>9}{:>8}".format(
            ev["gn"], ev["cg"], ev["sse"] ** 0.5, ev["psnr"],
            ev["psnr_held"], ev["elapsed"], ev["renders"], ev["backwards"]),
            flush=True)


if __name__ == "__main__":
    main()
