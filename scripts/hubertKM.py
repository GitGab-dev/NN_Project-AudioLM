import torch
import torch.nn as nn

from transformers import HubertModel
import joblib

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class SemanticTokenizer(nn.Module):
    def __init__(self, w2vCheckpointPath, kmeansCheckpointPath, chosenOutputLevel=9):
        """Initialization SemanticTokenize

        Args:
            w2vCheckpointPath (path): Path to HuBERT model checkpoint
            kmeansCheckpointPath (path): Path to KMeans model checkpoint
            chosenOutputLevel (int, optional): Output layer index. Defaults to 9.
        """

        super().__init__()

        self.embedder = HubertModel.from_pretrained(w2vCheckpointPath)
        self.kMeans = joblib.load(kmeansCheckpointPath)
        self.chosenOutputLevel = chosenOutputLevel

        self.kMeans.cluster_centers_ = self.kMeans.cluster_centers_.astype(float)

    def forward(self, x):
         
        embeddings = (
            self.embedder(x, output_hidden_states=True)
            .hidden_states[self.chosenOutputLevel]
            .squeeze()
        )

        normalizedEmbeddings = embeddings

        # Predict semantic tokens
        semanticTokens = self.kMeans.predict(normalizedEmbeddings)

        return semanticTokens, normalizedEmbeddings
