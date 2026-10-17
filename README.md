# Recently stopping everyday activities, death, and new limitation in activities of daily living: a multicohort study

This study relates recently stopping drinking, regular physical activity, and paid work to subsequent death and new limitation in activities of daily living among adults aged 60 years or older. Two consecutive interviews identify activities stopped and started; outcomes are assessed at the next interview, using a 22–30-month follow-up window. CHARLS, ELSA, HRS, KLoSA, and SHARE contribute to comparable-window analyses; MHAS contributes to wider-window analyses. Cohort-specific modified Poisson models use participant-clustered variance, with restricted maximum likelihood and safeguarded Hartung–Knapp pooling. Logistic-model risks are standardized to attainable populations. Additional analyses assess prior health, activity combinations, delayed landmarks, inclusion weighting, concurrent health, biomarkers, and discrimination beyond brief clinical information.

## Data access

Obtain the Gateway to Global Aging harmonized files and RAND HRS Longitudinal File from their custodians under the applicable access agreements. The analysis uses Harmonized CHARLS D2, ELSA H, KLoSA F, MHAS D, SHARE G, and RAND HRS 1992–2022 V1. Assay and interview supplements are required for the corresponding analyses. Individual-level data are not redistributed.

Set `HARMONIZED_ROOT` to the data release directory. The [universe configuration](config/behavioral_transition_outcome_scan_universe_v0.1.json) records cohort variables, source filenames, and input hashes. Its directory layout is:

```text
01_正式数据_分析用/{charls,elsa,hrs,klosa,mhas,share}.csv
02_变量字典_lookup/variable_lookup_long.csv
02_源数据备份/Harmonized源文件/{cohort}/primary/...
```

These are the analysis-ready long-form cohort files, variable lookup, and harmonized source files consumed by the loaders. Preserve the cohort directory names and supplemental file paths specified in [config](config). Source hashes identify the exact input releases.

Frailty-index analyses additionally require `derived_fi/{charls,elsa,hrs,share}-person-wave.parquet`, with `respondent_id`, `wave`, `fi_26`, and `fi_items_valid`, matching the [frailty configuration](config/behavioral_withdrawal_frailty_extension_v0.2.json). The index uses 32 candidate deficits, at least 26 observed, and disease, function, and cognition domains. The FI construction program is not included. FI models therefore require these derived files in addition to harmonized data; their final aggregate estimates are supplied.

## Layout

| Directory | Contents |
|---|---|
| [scripts](scripts) | Cohort loaders, multidomain models, context models, descriptive summaries, assay models, and postprocessors |
| [config](config) | Cohort definitions, model specifications, source filenames, and input hashes |
| [analysis/standardization](analysis/standardization) | Attainable-population risks, safeguarded pooling, and current-status support |
| [analysis/prediction](analysis/prediction) | Prediction and pre-window health |
| [analysis/composition](analysis/composition) | Activity combinations, losses, and gains |
| [analysis/sensitivity](analysis/sensitivity) | Outside-HRS comparisons, delayed landmarks, and inclusion weighting |
| [analysis/clinical](analysis/clinical) | Brief clinical baselines and discrimination |
| [results](results) | Cohort-level and pooled aggregate CSVs |
| [tables](tables) | Aggregate-only supplementary-table builders and generated tables |
| [figures](figures) | R figure code, aggregate input CSVs, and final PDFs |

## Environment and execution

The model-fitting environment is Python 3.12.4, numpy 2.2.6, pandas 2.2.2, scipy 1.14.0, and statsmodels 0.14.6. Figures use R 4.6.1 and ggplot2 4.0.3, with patchwork, svglite, ragg, and systemfonts. The R code uses Arial with a metrically compatible Liberation Sans substitution where configured by the system.

Use a Python 3.12.4 environment and install [requirements-lock.txt](requirements-lock.txt). [requirements.txt](requirements.txt) names the same model libraries. PyArrow supplies the Parquet reader for FI inputs; it does not define the fitted statistical models.

```bash
python -m pip install -r requirements-lock.txt
export HARMONIZED_ROOT="/path/to/harmonized-release"
bash scripts/run_core.sh
bash run_analyses.sh
```

Run these commands from the repository root. [run_core.sh](scripts/run_core.sh) fits the primary, context, opportunity, age, missingness, disease-specific, and assay models and produces descriptive summaries. [run_analyses.sh](run_analyses.sh) runs attainable standardization, prediction, activity composition, outside-HRS comparisons, delayed landmarks, inclusion weighting, and clinical baselines, in dependency order. Each runner writes aggregate outputs under `results`; person-level frames remain in memory. These commands require the corresponding private data inputs. Direct Python cohort runners expose their arguments with `--help` and accept `--data-root` or read `HARMONIZED_ROOT`.

With the derived FI inputs available, run the [FI cohort workflow](scripts/run_fi.sh). Then pool all cohort results:

```bash
bash scripts/run_fi.sh
python analysis/standardization/pool_hartung_knapp.py
```

The supplied aggregate CSVs also support running `pool_hartung_knapp.py` without access to individual data. Its [output](results/primary/hk_pooled.csv) contains point estimates, unmodified and safeguarded confidence intervals, and prediction intervals. The reported intervals use the columns ending in `_safeguarded`. [Standardized risks](results/standardization/pooled_risks_v0.1.csv) label the common population as `common_k_ge_2` and the category-specific population as `category_attainable`; model and standardization sample sizes are distinct.

Rebuild the fifteen supplementary tables from aggregate CSVs:

```bash
python tables/make_root_tables.py
python tables/build_group_2_3_6_7.py
python tables/build_9_12_13.py
python tables/build_tables_8_10_11_14.py
python tables/build_supplement.py
```

The last step checks every numeric cell against its CSV source and writes [table CSVs](tables/generated) and [cell-level provenance](tables/generated/supplement_provenance.csv). These commands require only the Python standard library.

Render the figures from their supplied aggregate input CSVs:

```bash
MS_ROOT="$PWD" Rscript figures/R/figures_gg.R
```

The outputs include [Figure 1](figures/out/figure1.pdf), [Figure 2](figures/out/figure2.pdf), and [Supplementary Figure S1](figures/out/supplementary_figure_s1.pdf). Figure 1 shows cohort-specific risk intervals; pooled-risk intervals and RR prediction intervals are reported in the tables.

## License

Code is distributed under the [MIT License](LICENSE). Cohort data remain subject to their custodians' terms.

## Citation

Li E, Kong L, Ren R, Liu H, Wang H. Recently stopping everyday activities, death, and new limitation in activities of daily living: a multicohort study. Under review.
