ad_finder
==============================

# Podcast Ad Finder
#### A way to detect and extract ads from any given podcast audio feed

## Table of Contents

* [Project Organization](#Project-Organization)
* [Installation](#Installation)
* [Create Data](#Create Data)
* [Project Organization](#Project-Organization)
* Installation
* Create Data
* Usage



## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models_             <- Trained and serialized models, model predictions
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    │── runs               <- Model Summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── data_create.py
    │   │   └── data_utils.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and make predictions
    │   │   ├── diarize_n_cluster.py <- diraize, cluster and segment ads using diarization module
    │   │   ├── encoder.py <- to train and eval the encoder for speaker diarization
    │   │   └── train_model_supervised.py <- to train and eval the supervised ad detection model
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Installation

```
python setup.py install 
```
or 

```
pip install -r requirements.txt
```
or 

```
pip install -e .
```

## Create Data

```
python src/data/data_create.py
```

## Usage

* Train and create speaker diarizations
    ```
    python src/models/encoder.py --help
    ```

* Segment and Extract ads using a speaker diarization module

    ```
    python src/models/diarize_n_cluster.py --help
    ```

* Train and classify ads using the supervised lstm model 

    ```
    python src/models/train_model_supervised.py --help
    ```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
