# ttHbb Dilepton Mass Reconstruction

Analytic mass reconstruction for the ttH(bb) dileptonic channel in CMS Run 3,
based on the Sonnenschein method. Built on top of PocketCoffea.

## What it does

For each event with two opposite-sign leptons and ≥4 jets, the code:

1. Loops over all (b, b̄, h1, h2) jet assignments from the 4 leading b-tagged jets
2. Scans a grid of top and W mass hypotheses
3. Solves the quartic polynomial for the neutrino momenta
4. Weights each solution by the parton-distribution probability (CT10)
5. Picks the best, second-best, third-best, and fourth-best combinations
   (with distinct Higgs pairings) and stores their reconstructed
   Higgs, top, W, and ttH masses

A DNN classifier and TRF-weight machinery are also integrated in the workflow.

## Main files

- `mass_reco_functions.py` — Sonnenschein quartic solver and neutrino
  reconstruction
- `custom_function.py` — selection and category helpers
- `workflow.py` — PocketCoffea processor (object preselection, DNN inference,
  TRF weights, mass-reco call)
- `params/` — calibration files, scale factors, working points
- `datasets_24/` — sample and dataset definitions for 2024 data and MC

## Dependencies

- PocketCoffea
- awkward, numpy, numba
- tensorflow (for DNN inference)
- parton (LHAPDF wrapper, for PDF weights)
- correctionlib

## Author

Polytimi Iosifidou — PhD candidate, CERN / CMS
