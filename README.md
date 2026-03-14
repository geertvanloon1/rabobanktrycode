# Replication Package for Dutch GDP Nowcasting with Stochastic Volatility and Factor Selection

## Overview

This repository contains the code used for our empirical analysis of Dutch GDP nowcasting with a Bayesian Dynamic Factor Model. The project is based on the framework of Zhang et al. (2022), which we adapt and apply to Dutch real-time data vintages.

The main goal of the project is to examine whether extending the standard Bayesian Dynamic Factor Model with stochastic volatility improves the real-time nowcasting accuracy of Dutch quarterly GDP. In addition, we study the role of factor selection by comparing fixed-factor and factor-selection model variants within the same nowcasting framework. 

The package is organized around three main empirical workflows:
- an in-sample analysis of model fit and latent factors,
- a monthly real-time nowcasting exercise comparing the baseline Bayesian model and its stochastic-volatility extension,
- and a real-time nowcasting exercise with factor selection enabled. 

The code follows a true-vintage setup: each nowcast is based only on the data that were actually available at the corresponding historical release date. Core model functions are stored in `src/`, while the scripts in `scripts/` are used to run the different empirical exercises and generate the plots and tables used in the report. 

## Repository structure

- `src/`  
  Building blocks of the model, including the code for data handling, model estimation, posterior sampling, nowcast construction, and result saving.

- `scripts/`  
  User-facing scripts that call the code in `src/` to run the empirical exercises and generate the plots and tables used in the report.

- `data/`  
  Input data files, including the Dutch real-time vintage files and the specification file `data/Spec_NL.xlsx`.

- `outputs/`  
  Saved run results, including posterior outputs, nowcast files, diagnostics, figures, and tables.


## Main workflows

### 1. In-sample workflow
Scripts:
- `10_insample_run.py`
- `11_insample_plot_diagnostics.py`
- `12_insample_bridge_diagnostics.py`
- `13_insample_diagnostic_crisis_backcasts.py`

This workflow produces the in-sample results of the paper.

### 2. Monthly real-time workflow
Scripts:
- `20_realtime_monthly_run.py`
- `21_realtime_monthly_run_parallel.py`
- `22_realtime_monthly_density_evaluation.py`
- `23_realtime_monthly_generate_artifacts.py`

This workflow produces the fixed-factor monthly real-time results of the paper.  
`20` and `21` are alternative versions of the same run, with `21` as the parallel version.

### 3. Real-time workflow with factor selection
Scripts:
- `40_realtime_selection_run.py`
- `41_realtime_selection_generate_artifacts.py`

This workflow produces the real-time results with factor selection.

### 4. Utility scripts
Scripts:
- `01_util_fix_vintage_columns.py`
- `02_util_select_transformations.py`

These scripts support the data preparation stage.

### 5. Legacy workflow not used in the final paper
Scripts:
- `30_realtime_release_run.py`
- `31_realtime_release_generate_artifacts.py`

These scripts are kept for reference but are not used in the final paper.

## Connection to the paper

- **In-sample results** come from scripts `10`–`13`.
- **Fixed-factor monthly real-time results** come from scripts `20`–`23`.
- **Factor-selection results** come from scripts `40`–`41`.
- Scripts `30`–`31` are not part of the final paper workflow.

## Data and specification files

Important inputs:
- `data/NL/` for the Dutch real-time vintage files
- `data/Spec_NL.xlsx` for the variable specification and transformations

## Output structure

Each run creates a dedicated folder under `outputs/`, containing:
- `nowcasts/`
- `posterior/`
- `diagnostics/`
- `figures/`
- `tables/`

## Source code structure

The folder `src/zhangnowcast/` contains the building blocks of the model. The code is divided into five main folders: `data/` for data preparation, `model/` for the model blocks, `inference/` for posterior estimation, `nowcast/` for nowcast construction, and `results/` for saving outputs.

### Source tree

src/zhangnowcast/
├── data/
│   ├── data.py
│   ├── load_data.py
│   ├── load_spec.py
│   └── zhang_buckets_from_vintages.py
├── inference/
│   ├── insamplesampler.py
│   ├── kalman_ffbs.py
│   ├── sampler.py
│   └── sv_pgbs.py
├── model/
│   ├── bridging.py
│   ├── factor_params.py
│   ├── gdp_block.py
│   ├── lkj_psi.py
│   └── selection.py
├── nowcast/
│   └── nowcast.py
└── results/
    └── io.py

### 1. Data preparation

The files in `src/zhangnowcast/data/` turn a real-time vintage file and the specification file into the data object that is used in estimation. This corresponds to the data setup in the report: a mixed-frequency dataset with monthly indicators, quarterly GDP as target variable, variable-specific transformations, standardization, and a real-time ragged-edge information set. 

#### `load_spec.py`
Role:  
Loads the specification file `data/Spec_NL.xlsx` and converts it into the ordered model specification used in the rest of the code.

Main functions:
- `load_spec(...)`: reads the specification sheet, keeps only the variables included in the model, extracts the series IDs, series names, frequencies, units, transformations, categories, and block structure, and sorts the variables by frequency.

Why this file matters:  
This file defines which variables enter the model and how they should be treated. In the report, this corresponds to the step where each indicator is assigned a transformation and frequency before estimation.

Connection to other files:  
The output of `load_spec.py` is passed directly to `load_data.py` and `data.py`.

#### `load_data.py`
Role:  
Reads one vintage Excel file and returns the transformed dataset in the order required by the model.

Main functions:
- `load_data(...)`: loads one vintage file, aligns its columns to the ordering from the specification file, applies the transformations, standardizes the series, and returns both the transformed data matrix and the raw aligned data.
- `read_data_excel(...)`: reads the vintage workbook itself and extracts the raw values, dates, and mnemonics.

Why this file matters:  
This file is where the raw vintage is turned into the transformed monthly dataset used by the model. In the report, this corresponds to the transformation and standardization step that makes the indicators approximately stationary and comparable in scale. 

Connection to other files:  
`data.py` calls `load_data(...)` to build the final estimation object.

#### `data.py`
Role:  
Builds the main `ZhangData` object that is passed into the sampler.

Main functions:
- `build_zhang_data(...)`: combines the transformed vintage data and the specification file into one structured object containing:
  - `X_m`: the monthly predictor panel,
  - `y_q`: quarterly GDP,
  - `dates_m` and `dates_q`,
  - `quarter_of_month` and `month_pos_in_quarter` for monthly-quarterly alignment,
  - `obs_idx` for the ragged-edge observation pattern.
- `slice_rolling_10y_window(...)`: trims the full data object to the rolling estimation window used in the real-time exercises.

Why this file matters:  
This is the key data-construction file. It translates the real-time vintage into the exact objects needed by the model: the monthly measurement panel, the quarterly GDP target, and the alignment structure that links monthly factor estimates to quarterly GDP through the bridge equation. It also keeps track of which indicators are observed in each month, which is essential for the ragged-edge real-time setup. 

Connection to other files:  
The `ZhangData` object built here is the direct input to `inference/sampler.py` and `inference/insamplesampler.py`.

#### `zhang_buckets_from_vintages.py`
Role:  
Handles the mapping from release dates and vintage dates to the real-time nowcast moments used in the scripts.

Main functions:
- `list_vintages(...)`: lists and dates the available vintage files.
- `pick_vintage_for_month_q(...)`: selects the correct vintage for a given month and release bucket.
- `infer_buckets_month_by_month(...)`: infers the typical release bucket of each series from the vintage files.

Why this file matters:  
This file supports the true-vintage nowcasting design. 

Connection to other files:  
These functions are mainly used by the run scripts to select the correct vintage before `build_zhang_data(...)` is called.

**Summary of the data folder:**  
The data folder starts from two inputs — a vintage file and `data/Spec_NL.xlsx` — and turns them into the `ZhangData` object that the samplers use. 

