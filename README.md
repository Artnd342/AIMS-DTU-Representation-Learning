# Latent Representation Learning via Self-Supervised Paradigms

A rigorous evaluation framework exploring how discriminative contrastive constraints and predictive reconstruction goals affect the underlying geometry of latent vector manifolds. Developed for the **AIMS DTU Research Internship Framework (2026)**.

## 🔬 Formal Mathematical Framework

### 1. SimCLR Contrastive Objective
SimCLR optimizes the **NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy Loss) across augmented pairs within a batch. For a positive pair $(i, j)$, the loss expression is formalized as:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ represents the cosine similarity and $\tau$ denotes the hyperparameter temperature.

### 2. Masked Autoencoder Objective
The Masked Autoencoder (MAE) processes an image by splitting it into patches, masking a large subset ($75\%$), and optimizing a pixel-level Mean Squared Error (**MSE**) reconstruction loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|M|} \sum_{i \in M} \|x_i - \hat{x}_i\|^2$$

Where $M$ is the set of masked patch indices, $x_i$ represents the original target pixel-patch value, and $\hat{x}_i$ is the predicted output sequence.

## 📊 Experimental Evaluation Summary

| Paradigm Architecture | Objective Classification | Clustering Silhouette Score | Dominant Latent Characteristic |
| :--- | :--- | :--- | :--- |
| **SimCLR** | Contrastive Discriminative | **~0.245** | Sharp, well-defined class boundaries, highly linear separable downstream. |
| **MAE** | Generative Reconstruction | **~0.082** | Continuous geometric manifolds; preserves internal structural contexts well. |

### Major Technical Blockers Resolved
* **Vanishing Collapse Solutions:** Early experiments showed representation collapse when utilizing weak augmentations. Introducing aggressive color jittering prevented the network from exploiting simple shortcut cues like background color histograms.
* **MAE Positional Alignment:** Patch recovery requires relative positional context. Incorporating custom grid coordinate sequences into the projection layer prevented spatial degradation during reconstruction.