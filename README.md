# GtdKO-V1-restoration

Analysis code for Cao et al., *Reactivation of photoreceptor inputs in adult mice leads to reconfiguration of cortical population activity geometry and restoration of behavioral vision*.

The code reproduces the analyses of the VR visual discrimination task (Figure 3M–N) and of V1 two-photon imaging (Figures 4–6, Supplementary Figures 3 and 4, and Supplementary Table 1).

Demixed subspace principal component analysis (dsPCA) is performed with the [dspca](https://github.com/ryhattori/dspca) package. A demo of dsPCA on V1 grating responses and the example pseudopopulation data are available in that repository:
- demo: [dsPCA_demo_Vision_Subspace.ipynb](https://github.com/ryhattori/dspca/blob/master/demo/dsPCA_demo_Vision_Subspace.ipynb)
- data: [v1_grating_pseudopopulation.npz](https://github.com/ryhattori/dspca/blob/master/data/v1_grating_pseudopopulation.npz)

## Requirements

- Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`)
- R 4.5 with ARTool, lme4 and emmeans (`Rscript install_r_packages.R`)

## Data

Data locations are set in `config.py` through environment variables:

| Variable | Contents |
|---|---|
| `V1_RAW_IMAGING` | one folder per imaging session (`<date>/<mouse>/<hemisphere>[/<depth>]`) with the stimulus log files |
| `V1_SUITE2P` | the same session folders with the Suite2p output (`suite2p/plane0`) |
| `V1_RESULTS` | folder for the per-session and group analysis results |
| `V1_BEHAVIOR_ZIP` | zip file with one trial-by-trial xlsx file per mouse for the VR task |

The sessions and groups are listed in `pipeline/session_analysis.py` and `pipeline/group_analysis.py` (`GROUPS`).

## Running the analyses

Run the scripts in the order below. Each script writes its outputs (CSV tables, statistics and figures) to its own folder, except the pipeline, which writes to `V1_RESULTS`.

| Step | Scripts | Figures |
|---|---|---|
| 1. Per-session analysis | `python pipeline/session_analysis.py [--parallel]` | responsive neurons, decoding and dsPCA of each session |
| 2. Group analysis | `python pipeline/group_analysis.py`<br>`python pipeline/shuffled_null_decoding.py` | Figure 6B–D; shuffled-label null for the decoding panels |
| 3. Behavior | `fig3_behavior/01`–`05` | Figure 3M–N |
| 4. Session metrics | `session_metrics/01`–`06` | metrics for Figures 4–6D; Supplementary Figure 3 |
| 5. Figures 4 and 5 | `fig4_fig5/01`–`03` | Figures 4 and 5 |
| 6. dsPCA-independent measures | `suppfig4_dspca_alternatives/01`–`03` | Supplementary Figure 4 |
| 7. Statistics table | `supp_table1/01`–`02` | Supplementary Table 1 |
| 8. Source data | `source_data/01`–`02` | source data files |

Python scripts are run with `python <script>.py` and R scripts with `Rscript <script>.R`, in numerical order within each folder.
