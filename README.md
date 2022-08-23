# Multi-dimensional antibiotic resistance time series reveals the phenotype-based fitness landscape for *Escherichia coli*

This repository contains the code used for the following paper:
> [1] **Multi-dimensional antibiotic resistance time series reveals the phenotype-based fitness landscape for *Escherichia coli***  
>Junichiro Iwasawa, Tomoya Maeda, Atsushi Shibai, Hazuki Kotani, Masako Kawada, Chikara Furusawa
>
> **Abstract:** The fitness landscape represents the complex relationship between genotype or phenotype and fitness under a given environment, the structure of which allows the explanation and prediction of evolutionary trajectories. Although previous studies have constructed fitness landscapes based on comprehensively studying the mutations in specific genes, the high dimensionality of genotypic changes prevents us from developing a fitness landscape capable of predicting evolution for the whole cell. Herein, we address this problem by inferring the fitness landscape for antibiotic resistance evolution by quantifying the phenotypic changes, that is, multi-dimensional time-series measurements of the drug resistance profile. Using the time-series data of drug resistance for multiple drugs, we inferred that the fitness landscape underlies the evolution dynamics of resistance. We showed that different peaks of the landscape correspond to different drug resistance mechanisms, thus supporting the validity of the inferred fitness landscape. We further discuss how inferred phenotype-fitness landscapes could contribute to the prediction and control of evolution. This approach bridges the gap between phenotypic/genotypic changes and fitness while contributing to a better understanding of drug resistance evolution.

## Dependencies

The full list of Python packages for the code is given in `requirements.txt`. These can be installed using:

```bash
pip install -r requirements.txt
```

## Brief description

The main figures (Fig.1–5) can be generated using 
`programs_for_figure_generation.ipynb`.

The supplemental figures can be generated using `Supplemental_figure_generation.ipynb`.

`src/fitness_landscape.py`, `src/time_series_generation.py` includes functions that were used in the jupyter notebooks above.

`data/` includes raw data for the experiments.
