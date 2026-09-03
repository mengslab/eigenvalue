# Eigenvalue Analysis v14.4.1

This release adds UI polish and a startup diagnostics panel while keeping the stable v14.4 analysis workflow unchanged.

## What's new
- startup diagnostics summary near the top of the app
- dedicated Diagnostics tab
- small button and layout polish
- clearer guidance for the top-left standalone **Run Analysis** workflow

## Run
```bash
cd eigenvalue_analysis_app_v14_4_1
./run.sh
```

# eigenvalue_analysis_app_v14_4_1

`eigenvalue_analysis_app_v14_4_1` is a stable release of an interactive computational biology app for analyzing rhythmic time-series data. The app provides a clean user interface for loading biological or synthetic time-series datasets, validating input structure, running rhythm-oriented analyses, comparing fitted components, and generating publication-style visual summaries.


## Overview

Biological time-series datasets often contain oscillatory structure arising from circadian, ultradian, metabolic, transcriptional, signaling, and reporter-based dynamics. However, these data are frequently noisy, short, irregularly sampled, or composed of multiple overlapping rhythmic components.

This application provides an integrated interface for exploring such signals through:

* data upload and validation
* representative demo datasets
* time-series visualization
* eigenvalue-inspired oscillatory decomposition
* fitted component visualization
* reconstructed signal summaries
* period, amplitude, phase, and decay-rate estimation
* publication-oriented figure panels
* clean Streamlit-based interactive analysis

The goal of this release is to provide a robust and easy-to-run application for exploratory rhythm analysis and figure preparation.

---

## Release status

**Version:** `v14_4_1`
**Release type:** Stable legacy foundation release
**Interface:** Streamlit
**Primary focus:** stability, usability, and reproducible local execution

This release is suitable for:

* local exploratory analysis
* manuscript figure preparation
* teaching biological rhythm analysis concepts
* testing synthetic rhythmic datasets
* developing future rhythm-analysis workflows

---

## Key features

### Stable app execution

This version was rebuilt to avoid common Streamlit package errors seen in earlier v14 versions, including missing module imports, missing session-state keys, absent startup scripts, and duplicated plot elements.

### Stand-alone quick analysis

The app includes a simplified quick-run workflow that allows users to rapidly execute representative analyses without navigating multiple nested panels.

### Clean user interface

The interface is organized around intuitive analysis steps:

1. upload or select input data
2. validate the time-series structure
3. configure analysis settings
4. run analysis
5. inspect fitted signals and decomposed components
6. export publication-oriented outputs

### Time-series data validation

The app checks whether the input dataset is compatible with downstream analysis, including:

* presence of valid time points
* numeric signal values
* missing values
* irregular sampling
* duplicate time points
* sufficient number of observations
* compatible column structure

### Oscillatory component analysis

The app summarizes oscillatory signals using interpretable parameters:

* mean level
* amplitude
* phase
* estimated period
* decay rate
* time of peak
* relative amplitude
* reconstructed signal

### Publication-oriented visualization

The app generates visual outputs designed for manuscript and grant preparation, including:

* fitted signal overlays
* decomposed oscillatory components
* reconstructed sum signals
* period-amplitude summaries
* component contribution plots
* clean benchmark/demo visualizations

---

## Example use cases

This app can be used to analyze:

* synthetic benchmark rhythmic signals
* circadian-like oscillations
* ultradian-like oscillations
* damped reporter signals
* bulk RNA-seq time-course summaries
* luciferase reporter trajectories
* GFP imaging-derived time-series signals
* multi-component biological rhythms
* noisy or partially irregular time-series data

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/eigenvalue_analysis_app_v14_4_1.git
cd eigenvalue_analysis_app_v14_4_1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

or, if a run script is included:

```bash
./run.sh
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

---

## Expected repository structure

```text
eigenvalue_analysis_app_v14_4_1/
├── app.py
├── requirements.txt
├── run.sh
├── README.md
├── LICENSE
├── data/
│   ├── demo_timeseries.csv
│   └── benchmark_examples.csv
├── outputs/
│   ├── figures/
│   └── tables/
├── assets/
│   └── app_preview.png
└── docs/
    ├── user_guide.md
    └── release_notes.md
```

Depending on your local package, some optional folders may not be present. The minimum required files are:

```text
app.py
requirements.txt
README.md
```

---

## Input data format

The app expects time-series data in tabular format, typically CSV or Excel.

A minimal input file should contain:

| time | signal |
| ---: | -----: |
|    0 |   1.02 |
|    2 |   1.41 |
|    4 |   1.76 |
|    6 |   1.55 |
|    8 |   0.93 |

For multi-signal datasets, columns may include multiple measured variables:

| time | gene_A | gene_B | reporter_1 | reporter_2 |
| ---: | -----: | -----: | ---------: | ---------: |
|    0 |   1.02 |   0.81 |        530 |        490 |
|    2 |   1.41 |   0.93 |        610 |        545 |
|    4 |   1.76 |   1.20 |        700 |        602 |

Recommended formatting:

* Time should be numeric.
* Signal columns should be numeric.
* Avoid duplicated time points.
* Use consistent time units, preferably hours.
* Missing values should be blank or encoded as `NA`.
* Replicate columns should be clearly named.

---

## Workflow

### Step 1: Load data

Users may upload a custom time-series file or select a representative demo dataset.

### Step 2: Validate input

The app checks time axis structure, missing values, and compatibility with rhythmicity analysis.

### Step 3: Configure analysis

Users can select or adjust analysis settings such as:

* sampling interval
* expected period range
* smoothing options
* number of components
* display options
* publication figure mode

### Step 4: Run analysis

The stand-alone run button executes the analysis workflow and stores the outputs in the active session.

### Step 5: Review results

The app displays:

* fitted trajectories
* decomposed oscillations
* reconstructed signal
* component parameters
* period and amplitude summaries
* visualization panels

### Step 6: Export outputs

Users can export or save:

* figures
* parameter tables
* fitted values
* reconstructed signals
* summary results

---

## Output parameters

The application reports the following rhythm-related parameters when available:

| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| Mean                 | Baseline or average signal level                 |
| Amplitude            | Magnitude of oscillatory component               |
| Phase                | Timing offset of the oscillation                 |
| Period               | Estimated cycle length                           |
| Decay rate           | Degree of damping or growth over time            |
| Time of peak         | Predicted peak timing                            |
| Relative amplitude   | Amplitude normalized to baseline or total signal |
| Reconstructed signal | Sum of inferred oscillatory components           |

---

## Representative outputs

Typical outputs include:

1. **Raw time-series plot**
   Displays the input signal over time.

2. **Fitted signal overlay**
   Compares observed and fitted rhythmic trajectories.

3. **Component decomposition plot**
   Shows individual oscillatory components.

4. **Reconstruction plot**
   Compares reconstructed signal to the observed signal.

5. **Parameter summary table**
   Lists estimated period, amplitude, phase, and decay parameters.

6. **Publication-style figure panel**
   Provides a clean multi-panel visualization suitable for manuscripts or presentations.

---

## Notes on interpretation

This app is designed for exploratory and methodological rhythm analysis. Estimated rhythmic parameters should be interpreted in the context of:

* sampling density
* time-course duration
* noise level
* biological replicate structure
* number of observed cycles
* waveform shape
* missing or irregular time points

For short or noisy datasets, period and phase estimates may be unstable. Biological interpretation should be supported by replicate-level validation and, when possible, independent experimental confirmation.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit'`

Install dependencies inside the active virtual environment:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

### `Could not open requirements file`

Make sure you are inside the correct folder:

```bash
cd ~/Downloads/eigenvalue_analysis_app_v14_4_1
ls
```

You should see:

```text
app.py
requirements.txt
```

### `No module named src`

This stable release is intended to be self-contained. If this error appears, check that you are using the correct `v14_4_1` package and not an earlier v14 build.

### Streamlit duplicate chart error

If Streamlit reports duplicated chart IDs, make sure each `st.plotly_chart()` or `st.pyplot()` call has a unique key when dynamically rendering repeated panels.

Example:

```python
st.plotly_chart(fig, use_container_width=True, key=f"plot_{name}_{i}")
```

### Session-state key error

If a session-state key is missing, initialize it before use:

```python
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
```

This release was designed to avoid this issue by initializing required session variables early in the app.

---

## Development notes

This release emphasizes:

* stable local execution
* simple dependency structure
* reduced nested imports
* clean Streamlit session-state initialization
* unique chart rendering keys
* user-friendly layout
* simplified quick-run workflow
* publication-focused output panels

It is intended as a foundation for future versions that may include expanded statistical testing, additional rhythm detection methods, benchmarking modules, and more advanced publication figure exports.

---

## Suggested citation

If you use this software in a manuscript or presentation, please cite the repository:

```text
Meng H. eigenvalue_analysis_app_v14_4_1: a framework for biological time-series rhythm analysis and oscillatory signal decomposition. GitHub. 2026.
```

For future IDRhythms-based versions, use the appropriate updated repository citation.

---

## License

Add your selected license here.


BSD-3 License
Copyright (c) 2026 Huan Meng
```

---

## Author

**Huan Meng, M.D., Ph.D.
huan.meng@gmail.com

GitHub: `https://github.com/mengslab`
Project: `eigenvalue_analysis_app_v14_4_1`

---

## Version history

### v14_4_1

Stable usability and configuration release.

Key updates:

* restored clean app startup
* stabilized dependency handling
* simplified module structure
* improved Streamlit session-state initialization
* improved quick-run workflow
* reduced redundant UI panels
* improved publication-style visualization organization
* prepared app as a stable foundation for future rhythm-analysis development

---

## Disclaimer

This software is provided for research and educational purposes. It is not intended for clinical diagnosis, medical decision-making, or regulatory use. Users are responsible for validating results in the context of their own datasets and experimental designs.
