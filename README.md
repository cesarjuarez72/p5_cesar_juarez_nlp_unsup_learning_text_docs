# Discover Topics in Text Using Unsupervised Learning (20 Newsgroups)

An end-to-end unsupervised Natural Language Processing (NLP) pipeline that transforms 18,846 unstructured forum posts into high-dimensional numerical feature spaces, automatically discovers latent macro-topics via $K$-Means clustering ($K=5$) without human labels, and visualizes topical convergence and geometric overlap using 2-dimensional Principal Component Analysis (PCA).

---

## Project Overview

- **Dataset:** 20 Newsgroups Corpus (`sklearn.datasets.fetch_20newsgroups`)[cite: 1, 2]
- **Preprocessing:** Stripped email headers, footers, and quote blocks to enforce semantic learning over metadata shortcuts[cite: 1, 2]
- **Feature Representation:** Term Frequency-Inverse Document Frequency (TF-IDF), $L_2$-normalized[cite: 1]
- **Vocabulary Parameters:** Top 5,000 features, English stop words excluded, minimum document frequency cutoff (`min_df=5`)
- **Clustering Algorithm:** Unsupervised K-Means ($K=5$, `init='k-means++'`, `n_init=10`, `random_state=42`)[cite: 1, 3]
- **Dimensionality Reduction:** 2D Principal Component Analysis (PCA) for geometric cluster visualization and overlap analysis[cite: 1, 7]

---

## Repository Structure

```text
.
├── notebooks/
│   └── Project5_CesarJuarez.ipynb     # Complete 2-cell cadence executable notebook
├── assets/
│   ├── cluster_distribution.png       # Cluster document distribution chart (K=5)
│   └── pca_clusters_2d.png            # 2D PCA cluster projection scatter plot
├── requirements.txt                   # Environment dependencies
├── LICENSE                            # Open-source license (MIT)
└── README.md                          # Production project documentation
```

---

## Machine Learning Pipeline Architecture

The implementation follows a strict 2-cell cadence per phase (Markdown Architecture & Analysis $\rightarrow$ Executable Python Code):

```
+-------------------------------------------------------------------------------+
| Part 1: Raw Text Ingestion (18,846 posts; headers, footers, quotes stripped) |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Part 2: TF-IDF Vectorization (5,000 features, min_df=5, English stop words)   |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Part 3: Unsupervised K-Means Partitioning (K=5, strict ground-truth isolation)|
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Part 4: Topic Discovery (Centroid keyword extraction & representative texts)  |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Part 5: 2D Dimensionality Reduction (PCA projection & cluster overlap analysis|
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Final Analysis: 7-Question Methodological & Technical Synthesis              |
+-------------------------------------------------------------------------------+
```

---

## Empirical Findings & Cluster Topics

The algorithm discovered five macro-themes across the corpus without ground-truth labels[cite: 1, 6]:

| Cluster ID | Discovered Macro-Theme | Document Count | Share (%) | Top Centroid Keywords | Representative Sample Source |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **Cluster 0** | Religion, Faith & Morality | 745 | 4.0% | `god`, `jesus`, `christ`, `believe`, `bible`, `faith` | `soc.religion.christian` (#9054) |
| **Cluster 1** | General Debate & Online Chatter | 9,670 | 51.3% | `just`, `like`, `edu`, `new`, `good`, `know` | Casual forum banter (#1107) |
| **Cluster 2** | Sports & Athletics | 1,172 | 6.2% | `game`, `team`, `games`, `year`, `hockey`, `players` | `rec.sport.hockey` FAQ (#16217) |
| **Cluster 3** | Computer Hardware & Systems | 3,559 | 18.9% | `thanks`, `windows`, `drive`, `card`, `know`, `file` | `comp.sys.mac.hardware` (#3646) |
| **Cluster 4** | Politics, Law & Social Opinion | 3,700 | 19.6% | `people`, `don`, `think`, `just`, `government`, `like`| `talk.politics.misc` (#654) |

[cite: 3, 6]

---

## Visualizations & Geometric Diagnostics

### 1. Document Allocation Across Clusters
The corpus exhibits an uneven distribution where **Cluster 1** functions as a conversational hub (51.3% of all posts), while technical and specialized domains (Sports, Religion, Computing) form distinct, concentrated clusters[cite: 3, 6, 7].

### 2. 2D PCA Coordinate Projection
- **Total 2D Explained Variance:** 1.02% (PC1: 0.61%, PC2: 0.41%)[cite: 7].
- **Geometry:** Documents form an open V-shaped projection:
  - **Cluster 3 (Computers):** Projects upward into the negative PC1 quadrant[cite: 7].
  - **Cluster 0 (Religion):** Projects upward into the positive PC1 quadrant[cite: 7].
  - **Cluster 2 (Sports):** Extends downward along the negative PC2 axis[cite: 7].
  - **Cluster 4 (Politics):** Occupies the intermediate mid-right coordinate space[cite: 7].
  - **Cluster 1 (General Chatter):** Anchors the vertex of the "V" around $(0.00, -0.03)$, acting as the linguistic hub[cite: 7].

---

## Technical Takeaways & Limitations

1. **TF-IDF Normalization:** Prevents long posts from dominating distance computations and penalizes high-frequency grammatical terms while elevating domain-specific terminology[cite: 1].
2. **Curse of Dimensionality:** Compressing 5,000 sparse word dimensions down to 2 axes retains ~1% of total variance, which explains the high density of points near the origin of the PCA plot[cite: 7].
3. **Bag-of-Words & Hard Partitioning Constraint:** K-Means assumes spherical cluster boundaries and requires single-cluster membership. It cannot capture polysemy, syntactic structure, or multi-topic posts (e.g., policy debates on computer encryption), highlighting the advantage of soft-clustering models (such as Latent Dirichlet Allocation) or transformer-based semantic embeddings for complex corpora[cite: 6].

---

## Setup & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/discovering-topics-20newsgroups.git
cd discovering-topics-20newsgroups
```

### 2. Configure Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Requirements (`requirements.txt`)
```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### 4. Run the Notebook
```bash
jupyter notebook notebooks/Project5_CesarJuarez.ipynb
```

---

## Author

- **Name:** Cesar Juarez
- **Program:** Data Analytics and AI
- **Coursework:** Unsupervised Machine Learning & Natural Language Processing

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
