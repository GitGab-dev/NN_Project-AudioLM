import torch
import torch.nn as nn

from transformers import HubertModel
import joblib

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class SemanticTokenizer(nn.Module):
    
    def __init__(self, w2vCheckpointPath, kmeansCheckpointPath, chosenOutputLevel = 9):

        super().__init__()
        
        self.embedder = HubertModel.from_pretrained(w2vCheckpointPath)
        self.kMeans = joblib.load(kmeansCheckpointPath)
        self.chosenOutputLevel = chosenOutputLevel

        self.kMeans.cluster_centers_ = self.kMeans.cluster_centers_.astype(float)
        
    def forward(self, x):
        embeddings = self.embedder(x, output_hidden_states = True).hidden_states[self.chosenOutputLevel].squeeze()
        
        normalizedEmbeddings = embeddings

        # Predict semantic tokens
        semanticTokens = self.kMeans.predict(normalizedEmbeddings)

        return semanticTokens, normalizedEmbeddings
    
def visualizeEmbeddings(embeddings, labels):
    
    #pca
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
    plt.title('PCA of embeddings')
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels)
    plt.title('t-SNE of embeddings')
    plt.show()