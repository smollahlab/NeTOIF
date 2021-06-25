NeTOIF
=====

This repository implements the NeTOIF for time-series omics data imputation and forecasting.

## Data

two datasets were used:

- RPPA (reverse phase protein array): The RPPA proteomics data were downloaded from the [Synapse platform](https://www.synapse.org/#!Synapse:syn12555331).
- GE (genome-wide gene expression): The GE data were published by [Mutarelli et al](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S2-S12).

## Requirement

  * Python 3.6
  * Tensorflow 1.14.0
  * Jupyter Notebook

## Usage

1. The imputation_task_RPPA_data.ipynb implements imputation task on the RPPA data.
2. The imputation_task_GE_data.ipynb implements imputation task on the GE data.
3. The forecasting_task_GE_data.ipynb implements forecasting task on the GE data.

To perform the imputation and forecasting tasks based on NeTOIF, please run the notebook step by step.

## Reference

[1] Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907. 2016 Sep 9.

## Citation

Shi, Min, and Shamim Mollah. "NeTOIF: A Network-based Approach for Time-Series Omics Data Imputation and Forecasting." bioRxiv (2021). doi: https://www.biorxiv.org/content/10.1101/2021.06.05.447209v1.abstract

