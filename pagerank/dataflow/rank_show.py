import sys
import numpy as np

def pagerank_show(output):
    output = np.fromfile(output, dtype=np.float32).reshape(-1)
    TOP_N = 20  # Number of top nodes to display
    # Get top N nodes
    top_indices = np.argsort(output)[-TOP_N:][::-1]  # Indices in descending order
    top_scores = output[top_indices]
    # Format and print results
    print(f"Rank | Node ID | Score")
    print("-" * 30)
    for rank, (node_idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"{rank:4d} | {node_idx + 1:7d} | {score / 1e6:.6f}")  # +1 converts to 1-based indexing


if __name__ == '__main__':
    pagerank_show("rank_result.bin")

