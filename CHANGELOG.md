# Changelog

A release packages the code; the plasma's settings are versioned on their own (`python xtorax.py versions`). Every release so far runs version 12 of the settings, so the whole-shot results in [VALIDATION.md](VALIDATION.md) hold for each of them.

## 1.2, 19 November 2026

The public repository. Whole-shot results unchanged: the plasma settings are version 12's.

- A workspace (`XTORAX_HOME`, the repository by default) and outside tools set by variables, in place of fixed paths.
- EPED-NN's weights and the reactor check's reference trace ship in `data/`.
- Both servers answer this machine only: a request whose Host is not 127.0.0.1 or localhost on the server's port, or a browser request from another site, gets 403. They refuse the decision kind `python`, which runs a named Python function, unless started with `--allow-python`.
- Decision models load from a file path (`examples/my_decision.py:make`).
- The quick start branches at 40 s, past the scenario's 25 s of fast current diffusion, and `examples/` has a decision model of your own with its session.
- The front end restyled after TORAX's own Plotly figures: Plotly's colours and Arial, plotly_dark by default and Plotly's light template from a Theme menu, Plotly's Plasma colour scale for the plasma.
- The 2D-solver scenarios removed (the solver is not part of the release), and the stress scenario lists only decision models that run from this repository.
- `core/tglfnn_bson.py`, which writes out TGLF-NN's spherical-tokamak networks for the transport option `tglfnn_st`, included.
- Fixes: `whole_shot_efit.py` and `mast_vde.py` stopped with an error when the workspace had no development-shot list; the scorer's figures went to `docs/figures` in the workspace and now go to `runs/twin/whole82/figures/`.

## 1.1, 9 November 2026

Exact branches, and pellets sized by the machine's injector. Whole-shot results unchanged: shot 20689 identical on all 46 scores.

**Branches.** In 1.0 a snapshot kept TORAX's state, the reactor's islands and ELM clock, the plant and the harness's own lists, but not the models a step calls, so a restored branch carried on with the last branch's coupled plasma: on 20689 a null branch run after a branch that fired a pellet, gas, an island and heating was in H-mode from its first step, because the fired branch was. An AI agent driving the harness found the transport table leaking the same way. A snapshot now also holds every model on the step (the coupled plasma's random draws, H-mode, collapse and density-loop state, the transport table, EPED-NN's schedule, the free boundary's client), the reactor's carried readings, the free boundary's geometry, the ELM draws, and FreeGSNKE's own state. Measured: on MAST 20689 with FreeGSNKE and on ITER with FreeGSNKE and EPED-NN, the null branch and a second restore repeat the first run exactly on every record (74 and 42 quantities, 40 steps each); a snapshot takes 0.01 s and a restore 0.01 to 0.04 s.

**Pellets.** The `pellet` command took a sphere's radius with a 3 mm default: 6.8e21 atoms, ITER's scale and about 50 times the electrons of a MAST plasma such as 20689's. Sizes now come from each machine's injector in `machines.py`: MAST's made cylinders 1.1, 1.35 and 1.7 mm long and wide, holding 0.6, 1.2 and 2.4e20 atoms, fired at 240 to 450 m/s from the top into the high-field side ([Valovič et al. 2008](https://iopscience.iop.org/article/10.1088/1742-6596/123/1/012039/pdf)); ITER's makes 5 mm fuelling and 3 mm ELM-pacing pellets. With no size given a pellet is the machine's default, `d_mm` takes an injector's size and `r_mm` a sphere's radius, and the answer gives the pellet's atoms and their share of the plasma's electrons, with a warning above one. On 20689 the 1.1 mm pellet raises the line density 47% at once and 21% after 10 ms, an e-folding of about 12 ms, inside MAST's measured retention of 2 to 20 ms in L-mode.

## 1.0, 9 November 2026

Version 12 of the plasma settings frozen and packaged: the reference set for the whole-shot results.
