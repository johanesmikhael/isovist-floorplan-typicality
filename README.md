# Latent Distribution: Measuring floor plan typicality with isovist representation learning   

This repository contains the code and models for Latent Norm Mapping, a novel method for analyzing floor plan datasets. The approach leverages Variational Autoencoders (VAEs) to capture spatial regularity and complexity, allowing for the identification of typical and atypical floor plans in a dataset. 
   
This work is derived from PhD research project "Machine Understanding of Architectural Space: From Analytical to Generative Applications" at EPFL Media and Design Lab (LDM).

**Link to thesis repo** > https://github.com/johanesmikhael/isovist-machine-understanding

## Repository Structure

```bash
├── vae/                                        # Variational autoencoder
├── floorplan_isovist_dataset/                  # Placeholder for the dataset (to be downloaded separately)
├── experiments/                                # Folder for experiment results
├── vae_train.py                                # Python script for vae training
├── 01_latent_norm_distribution_analysis.ipynb  # Notebook to evaluate latent norm typicality
├── 02_occlusivity_distribution_analysis.ipynb  # Notebook to evaluate occlusivity typicality
├── 03_variance_distribution_analysis.ipynb     # Notebook to evaluate variance typicality
├── requirements.txt                            # Python dependencies
└── README.md                                   # README
└── LICENSE                                     # MIT license
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/johanesmikhael/isovist-floorplan-typicality.git
cd isovist-floorplan-typicality
```
2. Create and activate a Conda environment:
```bash
# Create a new environment with Python 3.x (replace x with the specific version if needed)
conda create --name typicality-env python=3.8

# Activate the environment
conda activate typicality-env
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset for the experiments can be downloaded from Zenodo. Follow these steps:

Go to [Zenodo](https://doi.org/10.5281/zenodo.13871782) and download the dataset.
Unzip the dataset and place the content in the floorplan_isovist_dataset/ folder.


## Training and sampling
We can train the models by using the given script [model_name]_train.py and passing the suitable configuration via --config argument
```bash
# train a vae model
python vae_train.py --config ./vae/conf/vae_1000k.json
```

## Notebook for typicality analysis
The provided notebooks enable typicality analysis using Latent Norm, Isovist Occlusivity, and Isovist Variance

By default all the results will be stored in experiment/ folder.

## Keywords
isovist, floor plan, machine learning, architecture
