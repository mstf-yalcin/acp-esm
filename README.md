# ACP-ESM: A novel framework for classification of anticancer peptides using protein-oriented transformer approach

## Abstract
Anticancer peptides (ACPs) are a class of molecules that have gained significant attention in the
field of cancer research and therapy. ACPs are short chains of amino acids, the building blocks of
proteins, and they possess the ability to selectively target and kill cancer cells. One of the key ad-
vantages of ACPs is their ability to selectively target cancer cells while sparing healthy cells to a
greater extent. This selectivity is often attributed to differences in the surface properties of cancer
cells compared to normal cells. That is why ACPs are being investigated as potential candidates
for cancer therapy. ACPs may be used alone or in combination with other treatment modalities
like chemotherapy and radiation therapy. While ACPs hold promise as a novel approach to cancer
treatment, there are challenges to overcome, including optimizing their stability, improving
selectivity, and enhancing their delivery to cancer cells, continuous increasing in number of
peptide sequences, developing a reliable and precise prediction model. In this work, we propose
an efficient transformer-based framework to identify ACPs for by performing accurate a reliable
and precise prediction model. For this purpose, four different transformer models, namely ESM,
ProtBERT, BioBERT, and SciBERT are employed to detect ACPs from amino acid sequences.
To demonstrate the contribution of the proposed framework, extensive experiments are carried
on widely-used datasets in the literature, two versions of AntiCp2, cACP-DeepGram, ACP-740.
Experiment results show the usage of proposed model enhances classification accuracy when
compared to the literature studies. The proposed framework, ESM, exhibits 96.45% of accuracy
for AntiCp2 dataset, 97.66% of accuracy for cACP-DeepGram dataset, and 88.51 % of accuracy
for ACP-740 dataset, thence determining new state-of-the-art.

## Dataset
The numbers of Positive(ACPs) and Negative(non-ACPs) in the dataset are given below:

| Dataset           | Positive | Negative | Total |
|-------------------|----------|----------|-------|
| Anticp2 Alternate | 970      | 970      | 1940  |
| cACP-DeepGram     | 572      | 572      | 1144  |
| ACP-740           | 376      | 364      | 740   |


## Model Architecture

<p align="center">
  <img width="800px"  src="https://github.com/mstf-yalcin/acp-esm/assets/83976212/369637f9-821c-4c7e-a4a8-eba55aa5c65e">
</p>

