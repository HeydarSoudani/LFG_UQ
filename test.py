import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ["NDCG@3", "MRR@3", "Recall@3"]

retriever_scores = {
    "GIST-large (1024)":     [43.4, 40.8, 50.7],
    "BGE-large (1024)":      [52.1, 47.1, 66.7],
    "Codesage-small (768)":  [80,   77.1, 88.4],
    "SFR-Mistral (4096)":    [95,   94.2, 97.1],
}

x = np.arange(len(metrics))
n_retrievers = len(retriever_scores)
width = 0.8 / n_retrievers  # total group width ~0.8

# Use a matplotlib color set
cmap = plt.get_cmap("Set2")
colors = [cmap(i) for i in range(n_retrievers)]

plt.figure(figsize=(10, 3.5))

# Plot bars in a loop
for i, (name, scores) in enumerate(retriever_scores.items()):
    # Center groups around each metric position
    offset = (i - (n_retrievers - 1) / 2) * width
    plt.bar(x + offset, scores, width, label=name, color=colors[i])
    
plt.ylim(35, 101)   # change these numbers as you prefer

plt.xticks(x, metrics, fontsize=18)
plt.yticks([40, 60, 80, 100], fontsize=18)

# plt.ylabel("Score")
# plt.xlabel("Evaluation Metric")
# plt.title("Retriever Comparison on NDCG@3, MRR@3, Recall@3")
# plt.legend(fontsize=14)
# plt.tight_layout()

plt.legend(
    fontsize=15,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=True,           # turn box ON
    edgecolor="black"       # box border color
)

plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space at top for legend



# Save as PNG
plt.savefig("retriever_plot.png", dpi=800)
plt.savefig("retriever_plot.pdf", dpi=1200)


plt.show()
