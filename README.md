# A Kernel-Based Method for Accurate Steady-State Detection in Execution Time Series
Replication package of the work ‟*A Kernel-Based Method for Accurate Steady-State Detection in Execution Time Series* ”.

This work compares a new KB-KSSD method with the CP-SSD method introduced in *Towards effective assessment of steady state performance in Java software: Are we there yet?*.

All the relevant scripts are stored in `analysis-v2` folder together with all the generated plots, including the ones not included in the paper itself.
The rest of the repository is forked from the repository https://github.com/SEALABQualityGroup/steady-state as it was directly used for comparison.

## Requirements
All the requirements are given in the file `analysis-v2/environment.yml`.

To install the dependencies it is strongly advised to install all of them as a [Conda](https://www.anaconda.com/download) environment.

## RQ1: Method Comparisons
The method comparison results can be obtained as follows:

```
cd analysis-v2/man_steady_comparison
python3 compare_methods.py
```

## RQ2: Sobol's Sensitivity Analysis
The Sobol's analysis can be performed via running

```
cd analysis-v2/sensitivity_analysis_sobol
python3 sensitivity_sobol.py
```

It is recommended to run Sobol's analysis on a computational cluster, as the runtime can be ~72 hours.
