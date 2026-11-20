# Validation

The measured record behind [How far to trust it](README.md#how-far-to-trust-it). Everything here was measured unless it says inferred. The README gives the main table: version 12's errors on 25 whole MAST shots run hands-off, with how they were run and scored.

## Contents

- [Energies](#energies)
- [Profiles](#profiles)
- [Runs and events](#runs-and-events)
- [Repeat runs](#repeat-runs)
- [Across versions 9, 11 and 12](#across-versions-9-11-and-12)
- [What sets each error](#what-sets-each-error)
- [The error budget with the density held on the measured value](#the-error-budget-with-the-density-held-on-the-measured-value)
- [Options built, tested and not adopted](#options-built-tested-and-not-adopted)
- [Earlier checks against real data](#earlier-checks-against-real-data)
- [Rules the physics keeps](#rules-the-physics-keeps)

## Energies

Besides the stored energy as EFIT reads it (6.4%, in the README's table): the thermal energy against Thomson's kinetic energy, 21.6% per shot (signed -7%, -29 to +20); the thermal plus fast-ion energy against the energy EFIT's own beta_pol gives, signed +19% (-12 to +64; worst shot 240%), the fast-ion surplus described under [What sets each error](#what-sets-each-error).

## Profiles

Relative RMS over 0.1 < rho < 0.9 against Thomson (T_e, n_e) and charge exchange (T_i), median over shots, with holding the first frame in brackets:

| Profile | 5 ms | 20 ms | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|---|
| T_e | 9% (14) | 11% (27) | 17% (42) | 20% (58) | 21% (70) |
| T_i | 17% (30) | 29% (50) | 41% (80) | 33% (91) | 35% (95) |
| n_e | 12% (9) | 15% (16) | 15% (30) | 27% (47) | 14% (63) |

The simulation beats holding the start from about 20 ms on, on all three.

## Runs and events

All 25 runs reach the end of their window; one (18464) disrupts on its own 41 ms early, from too many fast ions at low density. Events over the time both ran (simulated; measured): ELMs 271; 237, sawtooth crashes 85; 107, share of time with an n = 1 mode 0.43; 0.31, share in H-mode 0.29; 0.10. Within a factor 2 of MAST, share of shots: ELMs 36%, crashes 56%, n = 1 mode time 44%, H-mode time 32%; H-mode on 8 shots that stayed in L-mode. By the readiness criteria written before the work, version 12 meets the harness criterion only: runs (one self-disruption), quantities (6 of 12) and events (under two thirds of shots) are not met.

## Repeat runs

The same shot, version and seed run twice give the same numbers on some shots and not on others. 20689 scored identically; 16458 matched to the fifth digit at the start and then drifted (q95 error 3.0% against 3.3%, outer edge 5.2 against 5.3 cm). Code, settings and inputs were the same and no timing enters any decision, so the drift is floating-point differences between runs (inferred). A difference between two options on one shot smaller than this is not a result.

## Across versions 9, 11 and 12

| Quantity | version 9 | version 11 | version 12 |
|---|---|---|---|
| line density | 22.4% | 22.5% | 13.0% |
| core T_e | 14.7% | 14.5% | 12.9% |
| whmd | 6.3% | 6.3% | 6.4% |
| q95 | 8.3% | 8.1% | 7.8% |
| elongation | 0.078 | 0.074 | 0.086 |
| axis radius | 2.26 cm | 2.04 cm | 2.32 cm |
| radiated power | 96% low | 88% low | 76% low |
| neutrons | 34.6% | 34.6% | 33.8% |
| runs not reaching their end or disrupting early | 1 | 2 | 1 |

## What sets each error

- **Density.** MAST had no density feedback: its valves ran pre-programmed waveforms, and its main inboard valve could not be shut during a shot, so its density follows the wall. The simulation holds the density on MAST's programmed line density where there is one, else on the start's Greenwald fraction; that programme itself sits a median 14% from MAST's density per shot (-62% on 22235, +30% on 20689), so holding it exactly cannot do better. Every published fuelling model keeps one number for how much gas a machine's wall keeps, which the rules below keep out of the general code.
- **Fast ions.** On MAST's measured profiles the beam model holds 2.04 times the fast-ion energy EFIT allows (EFIT's energy minus the kinetic), 1.69 times with the atomic physics rebuilt from published sources; neutrons from the fast ions present come to 1.15 and 0.91 of measured. This shows as the pressure-energy surplus (+19%), the outward axis (+1.4 cm), 18464's beta-limit collapse and the neutron scatter. What removes the rest is not settled: redistribution by MAST's modes (TRANSP needed about twice the anomalous fast-ion diffusion with MAST's long-lived mode; no published law sets the size) and charge exchange on edge neutrals (needs the wall).
- **H-mode.** The threshold is Martin 2008 with a low-aspect-ratio term from MAST, NSTX and Pegasus together; on MAST it gives 5 to 6 times Martin where MAST's own measurements give 8 to 10 times (Howlett 2023), so the simulation enters H-mode too readily. The shots it gets wrong sit inside the published fit's scatter (a factor 1.36 either way) 40 to 100% of the time: no published threshold decides them.
- **Radiated power.** MAST's prad_pol is a weighted sum over one midplane fan of 16 chords that sees the confined edge, the X-point regions and the inner divertor legs. The simulated reading sums the core (1 to 33 kW), carbon in the scrape-off layer and divertor (TORAX's extended Lengyel model with Eich's heat-flux width, 1 to 47 kW on most shots), the inner legs' recycled hydrogen (2 to 5 kW) and the main chamber's hydrogen when neutrals are run (11 to 81 kW). MAST's bolometers read 90 to 340 kW. The Lengyel term is steep: across its published inputs it spans a factor 30.
- **Shape.** The elongation sits 0.02 below EFIT's at the start and drifts to 0.10 below after 150 ms; the drift grows with how elongated MAST makes the plasma and does not follow the pressure excess. A lead, inferred and untested: FreeGSNKE's MAST coil description against EFIT's. The free boundary is forced up-down symmetric, so the vertical position is good only because these shots sat near the midplane.
- **Core temperature.** Unbiased in the median but scattered (-11 to +32% across shots); T_i profiles 33 to 41% after 50 ms. Core transport is TORAX's Bohm-gyroBohm held to published confinement laws; no transport network is valid at MAST's aspect ratio (TGLF-NN's spherical models were tested and did not help).
- **Sawteeth.** Porcelli's trigger with its thermal terms only. Crashes 85 against 107 measured: frequent real sawteeth on beam-heated shots are undercounted.
- **n = 1 modes.** Most of MAST's n = 1 activity is internal kinks near q_min = 1, which the simulation reads from its own q profile (q_min at or under 1.12 with beams on, Chapman et al. 2010). Its q_min tracks EFIT's moderately (correlation of mode share 0.45 across shots). What the mode does (rotation braking, fast-ion losses) is not modelled.
- **Starts.** FreeGSNKE's static solve at the start found the wrong plasma on 12 of 25 shots; it is now checked against EFIT and retried.

## The error budget with the density held on the measured value

Measured on version 11, before version 12 changed the density loop. With the line density held on Thomson's measured mean (an oracle run, not a mode for use): line density 4.3%, thermal energy 22 to 16%, radiated power 88 to 77% low; neutrons (34 to 33%) and core T_e (14.5 to 15.6%) barely move, and H-mode gets worse (time 0.42 against 0.09, false on 9 shots) because density lowers the threshold. So density alone would add one quantity; neutrons, temperature, H-mode and radiation need their own fixes.

## Options built, tested and not adopted

| Option | Measured | Why not |
|---|---|---|
| Beams rebuilt from published atomic physics (`beam_options`: Suzuki stopping, halo charge exchange, the plasma's Coulomb logarithm, orbit averaging) | fast-ion energy on measured profiles 2.04 to 1.69 times EFIT's; neutrons 33.8 to 30.3% | absorbs more power, moves the axis out (2.3 to 3.1 cm), more self-disruptions, sawtooth chatter |
| Each shot's measured beam energy split (`species: "measured"`) | old shots' neutrons nearer (16148: 1.86 to 1.37 of measured) | adopted only inside the rebuilt beams, which were not |
| Fuelling by neutrals from MAST's measured wall pressure (`fuelling: kn1d_gauge`, KN1D) | L-mode shots: line density -8 to -23%; H-mode shots: +77 to +330% | an oracle; in H-mode the edge source piles up to ten times MAST's edge density |
| The scrape-off layer's own particle balance (`sol: balance`) | separatrix 0.7 of the line average at a D_perp of 1 m^2/s; the core's density doubled in 25 ms | MAST's measured D_perp (1 to 5 m^2/s) untested |
| ELMs taking particles out with the density pin off | no change on the H-mode overfuelling | the edge shelf, not the ELMs |

## Earlier checks against real data

| What | Real data | Result |
|---|---|---|
| Profiles, replay with the edge given | JET, 20 discharges | T_e 12 to 19%, T_i 14 to 22%, n_e 12 to 18%, stored energy 9 to 17% at 0.5 to 2 s; C-Mod worse, T_e up to 32% |
| Profiles, hands-off | JET, C-Mod | JET T 20 to 38%; C-Mod fails on 2 of 5 |
| Response to a change in heating | JET, 20 starts; MAST, 17 beam steps | JET T_e 15 to 19% with the edge given (holding 43 to 55%); MAST step up T_e 13% at 50 ms (holding 27%) |
| Island growth and width | MAST, 418 n = 1 and 104 3/2 modes | median width 0.73 to 1.10 of real; little skill on which grows (AUC 0.57 to 0.65) |
| Island model, version 5 | MAST n = 1; ITER against published numbers | width distribution error 0.23 to 0.32; ITER seed to lock 1.5 to 1.8 s (published 1.4 to 2.4), saturated width 34 to 35 cm (published 32) |
| Time from onset to lock, given the real width | MAST, 418 modes | 18 against 39 ms: 2.2 times too fast |
| Sensors from the simulated plasma | MAST pickup coils | coil places right to about 5 cm, readings within 4 to 14% on real maps |
| Which way MAST's plasma goes after the thermal quench | 40 real disruptions | 12 of 13 where the collapse clearly moves the equilibrium |
| DREAM after the onset | Vallhagen et al. 2020, three published cases | reproduced; ITER runaways 3.1 to 7.3 MA whatever the timing |
| The most extreme real shots | MAST, 92 shots | the worse profile channel 1.4 to 1.6 times its typical error; nothing runs away |

## Rules the physics keeps

No law fitted to MAST enters the general code: coefficients come from published multi-machine work or from physics, a machine file holds hardware and programmes only, and MAST's measurements may check a model, never set it. Three fixes are ruled out by that rule and left open: a wall number per machine, MAST's measured L-H threshold in its machine file, and the size of fast-ion redistribution by MAST's modes. Versions are frozen: a change becomes a new version only after it has been scored on all 25 shots, compared with the recommended version on the same starts, and found better.
