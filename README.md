# Poodle
<ins>**P**</ins>rojecting <ins>**O**</ins>bservations <ins>**O**</ins>n a <ins>**D**</ins>eep <ins>**L**</ins>earned <ins>**E**</ins>mbedding

## Background
Clustering techniques that use deep learned embeddings often outperform conventional clustering techniques such as k-means. However, when it comes to projecting new samples onto the learned embedding there is a lack of guidelines & tools. We built POODLE to facilitate the projection of new samples onto this product space. Samples are clustered one-by-one according to their orientation in the latent space.

## Fusion-based
In case the replication set is missing any variables, we build a shared product space and only project patients on the variables present in both sets.

## Deep learning technique
We used the autoencoder architecture of MAUI as an example. However, one could also adopt a different deep learning architecture or even a factor analysis technique (like MOFA). Currently, this github repo does not provide examples for other techniques.


