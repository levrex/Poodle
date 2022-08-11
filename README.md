# Poodle
<ins>**P**</ins>rojecting <ins>**O**</ins>bservations <ins>**O**</ins>n a <ins>**D**</ins>eep <ins>**L**</ins>earned <ins>**E**</ins>mbedding

## Background
Clustering techniques that use deep learned embeddings often outperform conventional clustering techniques such as k-means [1] (https://www.nature.com/articles/s41598-021-91297-x). However, when it comes to projecting new samples onto the learned embedding there is a lack of guidelines & tools. We built POODLE to facilitate the projection of new samples onto this product space. Samples are clustered one-by-one according to their orientation in the latent space.

## Deep learning technique
We used the autoencoder architecture of MAUI as an example. However, one could also adopt a different deep learning architecture or even a factor analysis technique (like MOFA). Currently, this github repo does not provide examples for other techniques.

## Robust to difference in dimensionality
Poodle is flexible for situations where certain data is absent in the clinic, as one may build a shared product space and only project patients on the variables present in both sets. However, ensure that the key features are still included.

## How to start
Start a notebook session on your device and open the following file :  
[Start here](examples/projecting_patients.ipynb) 

## WIP
Be aware that this github repo is still a work in progress. We will update the readme as we make new additions to the tool. For example: we aim to add tSNE projection, baseline comparison and batch correction in the near future.