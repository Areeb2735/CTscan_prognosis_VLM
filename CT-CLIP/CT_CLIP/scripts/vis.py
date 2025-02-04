import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np

embeddings_all = np.load("/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/embeddings_new_TNM.npy", allow_pickle=True).item()

# Extract labels and embeddings
labels = list(embeddings_all.keys())  # Patient IDs
# embeddings = np.array([embeddings_all[label]["text_embedding"] for label in labels])  # Image embeddings
embeddings = np.array([np.concatenate([embeddings_all[label]["text_embedding"], embeddings_all[label]["image_embedding"]], axis=0) for label in labels])

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)  # Adjust perplexity based on dataset size
tsne_results = tsne.fit_transform(embeddings[0:50])

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=label)

# Annotate points with labels
for i, label in enumerate(labels):
    plt.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]))

plt.title("t-SNE Visualization of Text Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
# plt.legend()
# plt.show()
plt.savefig('tsne_both.png')
