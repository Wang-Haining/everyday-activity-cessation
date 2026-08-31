# Recent cessation of everyday activities and near-term death and loss of independence in older adults

Code and frozen aggregate results for a coordinated multicohort study of six
harmonised ageing cohorts: CHARLS, ELSA, HRS, KLoSA, MHAS and SHARE. Five
contribute to the primary analysis, over a comparable 22 to 30 month outcome
window; MHAS has no interval in that window and is reported over its own wave
spacing. Under review.

## Outcomes

Death is the primary outcome. New limitation in activities of daily living, new
limitation in instrumental activities of daily living and multimorbidity
progression are secondary.

The cohort models in `scripts/` fit a wider outcome set than the paper reports,
and the frozen tables in `artifacts/` therefore carry estimates for four newly
reported diagnoses as well. Those are not omitted results: the graded count was
evaluable for them in two cohorts, below the prespecified minimum of three for a
cross-cohort estimate, so they could only ever carry the binary any-cessation
exposure and are reported separately. The model code is left as it ran, because
it is what produced the tables here.

## Layout

    scripts/    cohort models; run where the data are
    artifacts/  the aggregate tables those runs produced, frozen
    analysis/   turns the frozen aggregates into the reported estimates and figures
    config/     design decisions, fixed before any outcome model was fitted,
                including the source item, recall period and threshold behind
                every harmonised variable
    tests/      unit tests for the frailty and eligibility logic

`requirements.txt` gives the ranges the code is written against;
`requirements-lock.txt` and `figures/R/sessionInfo.txt` pin the exact Python and
R environments the reported models and figures were produced in, which are the
versions the manuscript names.

Aggregate only. The largest file in `artifacts/` is 496 rows of model output,
there are no identifiers, and no code path here can write one. Run manifests
keep the SHA-256 of every input file, with filesystem paths removed, and the
`analysis_id` they record is the internal release that produced that table.

## Reproducing the reported numbers

Needs no data access.

    pip install -r requirements.txt

    python analysis/make_multidomain_manuscript_displays.py   # main tables and panels
    python analysis/make_clinical_displays.py                 # work-exit panel and the risk panels
    python analysis/make_reference_figure.py                  # comparison groups
    python analysis/leave_one_cohort_out.py                   # re-synthesis dropping each cohort
    python analysis/competing_risk_recount.py                 # competing-risk recount
    python analysis/assemble_figures.py                       # composites from the panels
    python analysis/make_consort_drawio.py                    # the study flow diagram

## Drawing the three manuscript figures

They are drawn in R, from tables the Python writes out of the same frozen
artifacts, so nothing in the figures can disagree with the reported numbers.

    python analysis/export_figure_data.py     # figures/data/*.csv, row counts asserted
    Rscript figures/R/figures_gg.R            # needs ggplot2, patchwork, svglite, ragg
    python analysis/check_r_figures.py        # every figure at its declared size

`figures/R/figures_gg.R` reads `MS_ROOT` and defaults to the working
directory, so it runs wherever the R packages are. `scripts/build_lancet_figures.sh`
is the convenience wrapper the authors used to run it on a cluster; set `HOST`
and `REMOTE_DIR` for your own, or ignore it and call Rscript directly.

`leave_one_cohort_out.py` is the quickest integrity check: its "none omitted"
rows must reproduce the published estimates exactly, and they are computed from
the per-cohort log-estimates and standard errors by the same REML and
Hartung-Knapp routine used for the primary synthesis.

## Rerunning the cohort models

`scripts/` needs the harmonised source files, which are not in this repository.

    ROOT=/path/to/analysis sbatch scripts/run_multidomain_behavioral_withdrawal_models.sbatch

Each study releases its own data under registration and a data-use agreement:
CHARLS (charls.pku.edu.cn), ELSA (elsa-project.ac.uk), HRS
(hrsonline.isr.umich.edu), KLoSA (survey.keis.or.kr), MHAS (mhasweb.org) and
SHARE (share-eric.eu). The harmonised versions and codebooks come from the
Gateway to Global Aging Data (g2aging.org); HRS additionally uses the RAND HRS
Longitudinal File.

## License

MIT. Code only. It grants no access to any cohort's data.
