import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import os

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    import gn_solve

    return Image, gn_solve, io, mo, np, os, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Gauss-Newton appearance solve

        Geometry is frozen from a checkpoint; the unknowns are the appearance coefficients
        `texel_sv_rgb`. The render is **piecewise affine** in them
        (`powerfoam/color_fn.py:95-98`: an affine blend, `+0.5`, then a ReLU), so fitting
        appearance is a least-squares problem and Gauss-Newton applies.

        `J v` is a central difference of the forward render; `J^T u` is the existing backward
        kernel via autograd. Inner CG on the normal equations, outer Levenberg-Marquardt.

        **Watch the two PSNR curves.** *fit* is the views being solved; *held* is views the
        solver never sees. If *held* diverges from *fit*, the system is under-constrained and
        the solve is memorising directions rather than recovering appearance.
        """
    )
    return


@app.cell
def _(mo, os):
    def _find(pattern_dir, suffix):
        out = []
        if os.path.isdir(pattern_dir):
            for d in sorted(os.listdir(pattern_dir)):
                p = os.path.join(pattern_dir, d, suffix)
                if os.path.exists(p):
                    out.append(os.path.join(pattern_dir, d))
        return out

    runs = _find("output", "model.pt")
    default_run = next(
        (r for r in runs if "truefrozen" in r and "0062" in r), runs[0] if runs else ""
    )

    run_sel = mo.ui.dropdown(
        options=runs, value=default_run, label="checkpoint (output/<run>)"
    )
    views = mo.ui.slider(2, 40, value=8, label="fit views", show_value=True)
    held = mo.ui.slider(0, 20, value=12, label="held-out views", show_value=True)
    gn_iters = mo.ui.slider(1, 30, value=8, label="GN steps", show_value=True)
    cg_iters = mo.ui.slider(1, 40, value=15, label="CG iters per GN step", show_value=True)
    lam = mo.ui.number(1e-9, 1.0, value=1e-4, step=1e-5, label="LM damping lambda")
    fd_eps = mo.ui.number(1e-4, 1.0, value=1e-2, step=1e-3, label="FD step for Jv")

    mo.vstack(
        [
            run_sel,
            mo.hstack([views, held], justify="start"),
            mo.hstack([gn_iters, cg_iters], justify="start"),
            mo.hstack([lam, fd_eps], justify="start"),
        ]
    )
    return cg_iters, fd_eps, gn_iters, held, lam, run_sel, views


@app.cell
def _(mo):
    build_btn = mo.ui.run_button(label="1. Load scene + checkpoint")
    build_btn
    return (build_btn,)


@app.cell
def _(build_btn, fd_eps, gn_solve, held, lam, mo, os, run_sel, views):
    mo.stop(not build_btn.value, mo.md("*Press **Load scene + checkpoint** to begin.*"))

    with mo.status.spinner(title="Loading scene, images and checkpoint..."):
        solver = gn_solve.GNSolver(
            os.path.join(run_sel.value, "config.yaml"),
            os.path.join(run_sel.value, "model.pt"),
            n_views=views.value,
            n_held=held.value,
            lam=lam.value,
            fd_eps=fd_eps.value,
        )
        aff = solver.affinity()

    _aff_rows = "\n".join(
        "| {:g} | {:.3e} | {} |".format(
            s, v, "locally AFFINE" if v < 1e-2 else "ReLU clamps active"
        )
        for s, v in aff.items()
    )

    mo.md(
        """
        **{prims:,}** primitives &nbsp;&nbsp; **{unk:,}** unknowns &nbsp;&nbsp;
        **{nfit}** fit views, **{nheld}** held-out (of {navail})

        | perturbation scale | relative nonlinearity | verdict |
        |---|---|---|
        {aff}

        SGD baseline (this checkpoint, 30k steps): **{bfit:.3f} dB** on fit views,
        **{bheld:.3f} dB** on held-out.
        """.format(
            prims=solver.n_primitives,
            unk=solver.n_unknowns,
            nfit=len(solver.cams),
            nheld=len(solver.hcams),
            navail=solver.n_views_avail,
            aff=_aff_rows,
            bfit=solver.base_psnr,
            bheld=solver.base_held,
        )
    )
    return (solver,)


@app.cell
def _(mo, solver):
    mo.stop(solver is None)
    solve_btn = mo.ui.run_button(label="2. Run the solve")
    solve_btn
    return (solve_btn,)


@app.cell
def _(Image, cg_iters, gn_iters, io, mo, np, plt, solve_btn, solver):
    mo.stop(not solve_btn.value, mo.md("*Press **Run the solve** to start.*"))

    def _png(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def _figure(gn_hist, cg_hist):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        if gn_hist:
            xs = [h["cg"] for h in gn_hist]
            ax[0].plot(xs, [h["psnr"] for h in gn_hist], "o-", label="GN, fit views")
            if not np.isnan(gn_hist[-1]["psnr_held"]):
                ax[0].plot(
                    xs, [h["psnr_held"] for h in gn_hist], "s-", label="GN, held-out"
                )
        ax[0].axhline(solver.base_psnr, ls="--", c="k", lw=1, label="SGD 30k, fit")
        if not np.isnan(solver.base_held):
            ax[0].axhline(
                solver.base_held, ls=":", c="gray", lw=1, label="SGD 30k, held-out"
            )
        ax[0].set_xlabel("cumulative CG iterations")
        ax[0].set_ylabel("PSNR (dB)")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)

        if cg_hist:
            ax[1].semilogy(
                [c["cg"] for c in cg_hist], [c["resid"] for c in cg_hist], lw=1
            )
        ax[1].set_xlabel("cumulative CG iterations")
        ax[1].set_ylabel("||r|| of the normal equations")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        return fig

    gn_hist, cg_hist = [], []
    total = gn_iters.value * cg_iters.value + gn_iters.value

    with mo.status.progress_bar(total=total, title="Gauss-Newton") as bar:
        for ev in solver.steps(gn_iters.value, cg_iters.value):
            bar.update(1)
            if ev["kind"] == "cg":
                cg_hist.append(ev)
                if ev["cg"] % 3:
                    continue
            else:
                gn_hist.append(ev)

            fig = _figure(gn_hist, cg_hist)
            blocks = [mo.as_html(fig)]
            plt.close(fig)

            if ev["kind"] == "gn" and ev["gn"] > 0:
                gt, gn_img, sgd_img = solver.preview(0)
                blocks.append(
                    mo.hstack(
                        [
                            mo.vstack([mo.md("**ground truth**"), mo.image(_png(gt))]),
                            mo.vstack(
                                [
                                    mo.md("**Gauss-Newton, step {}**".format(ev["gn"])),
                                    mo.image(_png(gn_img)),
                                ]
                            ),
                            mo.vstack(
                                [mo.md("**SGD, 30k steps**"), mo.image(_png(sgd_img))]
                            ),
                        ],
                        widths="equal",
                    )
                )

            rows = "\n".join(
                "| {gn} | {cg} | {sse:.4e} | {psnr:.3f} | {held:.3f} | {lam:.2e} |"
                " {acc} | {el:.1f} |".format(
                    gn=h["gn"],
                    cg=h["cg"],
                    sse=h["sse"] ** 0.5,
                    psnr=h["psnr"],
                    held=h["psnr_held"],
                    lam=h["lam"],
                    acc="yes" if h.get("accepted", True) else "**halved**",
                    el=h["elapsed"],
                )
                for h in gn_hist
            )
            blocks.append(
                mo.md(
                    "| GN | CG | sqrt(SSE) | PSNR fit | PSNR held | lambda |"
                    " accepted | s |\n|---|---|---|---|---|---|---|---|\n" + rows
                )
            )
            mo.output.replace(mo.vstack(blocks))

    final = gn_hist[-1] if gn_hist else None
    return final, gn_hist


@app.cell
def _(final, mo, solver):
    mo.stop(final is None)
    _verdict = (
        "The held-out gap is small: the solve is recovering appearance."
        if abs(final["psnr"] - final["psnr_held"]) < 2.0
        else "**The held-out gap is large: the system is under-constrained at this view "
        "count.** `texel_sv_rgb` holds sv_dof x num_texel_sites x 3 direction-dependent "
        "coefficients per cell, so few views cannot pin them down. Raise *fit views* "
        "before reading anything into the fit-view number."
    )
    mo.md(
        """
        ### Result

        | | fit views | held-out |
        |---|---|---|
        | SGD, 30k steps | {bf:.3f} dB | {bh:.3f} dB |
        | Gauss-Newton, {gn} steps ({cg} CG, {t:.0f} s) | {f:.3f} dB | {h:.3f} dB |

        {verdict}
        """.format(
            bf=solver.base_psnr,
            bh=solver.base_held,
            gn=final["gn"],
            cg=final["cg"],
            t=final["elapsed"],
            f=final["psnr"],
            h=final["psnr_held"],
            verdict=_verdict,
        )
    )
    return


if __name__ == "__main__":
    app.run()
