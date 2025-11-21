# HGOT-SER
This is an implementation of HGOT-SER (Hybrid Genetic Optimization for Multi-Granularity Transformer-based SER) model.

# Datasets

IEMOCAP (English). Four high-frequency emotion categories are selected: anger (1,103), neutral (1,708), happiness (1,636), and sadness (1,084), yielding 5,531 valid utterances. 

CASIA (Chinese) contains 1,200 utterances recorded by 4 professional actors (2 male, 2 female), covering six emotions: anger, sadness, happiness, neutral, surprise, and fear. Each emotion category includes 200 utterances.

EMODB (German) includes 535 utterances by 10 professional speakers (5 men and 5 women), covering seven emotions: anger, sadness, fear, boredom, neutral, disgust, and happiness. 

# Speech proprecessing

All audio clips are resampled to 16 kHz.

The exact train/validation/test splits used in our paper are fully provided in:

1. IEMOCAP     - preprocess  / 

2. CASIA       - pro_CASIA /

3. EMODB      - pro_EMODB /

Note: The dataset split is random. Running the script will produce different splits,
but the splitting procedure is identical to our experimental setting.

# Model

The full model implementation will be released after the review process.

Currently, we provide dataset splits and preprocessing steps to ensure reproducibility without revealing proprietary components.

The required libraries and their version:

Python 3.8

PyTorch 1.12.0
