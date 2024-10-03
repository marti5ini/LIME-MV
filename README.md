# Handling Missing Values in Local Post-Hoc Explainability

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://pypi.org/project/biasondemand) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code for the paper titled *Handling Missing Values in Local Post-Hoc Explainability*, an extension of post-hoc explainability approaches to managing missing values in decision-making processes with tabular data.

Check out the paper here: [[pdf](https://link.springer.com/chapter/10.1007/978-3-031-44067-0_14)]

## Abstract

Missing data is a common issue in real-world scenarios when Artificial Intelligence (AI) systems are used for decision-making with tabular data. Effectively managing this challenge is crucial for maintaining AI performance and reliability. While some machine learning models can handle missing data, there is a lack of post-hoc explainability methods that address this challenge.

This repository introduces a novel extension to a widely used local model-agnostic post-hoc explanation approach. The method integrates state-of-the-art imputation techniques directly into the explanation process, allowing users to understand not only feature importance but also the role of missing values in the prediction. Extensive experiments demonstrate the superior performance of this method compared to traditional imputation-based approaches.

## Setup

To set up the project, you'll need Python version >=3.8 and the required libraries listed in the `requirements.txt` file. It's highly recommended to use a virtual environment for managing dependencies.

1. Clone the repository:

   ```
   git clone https://github.com/marti5ini/LIME-MV.git
   cd LIME-MV
   ```

2. Install dependencies in a virtual environment:

    ```
   python3 -m venv ./venv  # Optional but recommended
    source ./venv/bin/activate  # For Linux/macOS
    .\venv\Scripts\activate  # For Windows
    pip install -r requirements.txt

   ```

## Citation

If you use `LIME-MV` in your research, please cite our paper:

```
@inproceedings{cinquini2023handling,
  title={Handling missing values in local post-hoc explainability},
  author={Cinquini, Martina and Giannotti, Fosca and Guidotti, Riccardo and Mattei, Andrea},
  booktitle={World Conference on Explainable Artificial Intelligence},
  pages={256--278},
  year={2023},
  organization={Springer}
}
