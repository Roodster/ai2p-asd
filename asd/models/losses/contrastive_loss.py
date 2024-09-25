import torch.nn as nn
import torch as th
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
    
    def forward(self, outputs):
        """
        Args:
            first_transformed: Tensor of shape [batch_size, features] from transformation 1 (e.g. P(T1(x)))
            second_transformed: Tensor of shape [batch_size, features] from transformation 2 (e.g. P(T2(x)))

        Returns:
            contrastive_loss: Computed contrastive loss as per equation (5)
        """
        first_transformed, second_transformed = outputs
        # Compute the cosine similarity between the transformed pairs
        pos_similarity = self.similarity(first_transformed, second_transformed)
        
        # Organize batch into positive and negative pairs, including all-to-all comparisons
        batch_size = first_transformed.size(0)
        
        # Calculate the full similarity matrix for the transformed features
        combined_transformed = th.cat([first_transformed, second_transformed], dim=0)  # [2*batch_size, features]
        
        print('combined: ', combined_transformed)
        similarity_matrix = self.similarity(combined_transformed.unsqueeze(1), combined_transformed.unsqueeze(0))  # [2*batch_size, 2*batch_size]
        print('similarity_matrix: ', similarity_matrix)

        # Mask out the diagonal (self-similarity)
        mask = th.eye(2 * batch_size, device=similarity_matrix.device).bool()
        print('mask: ', mask)


        similarity_matrix.masked_fill_(mask, float('-inf'))

        # Split similarity matrix into positive and negative pairs
        positive_similarities = pos_similarity / self.temperature
        print('positive_similarities: ', positive_similarities)

        # Negative pairs: log-sum-exp trick
        neg_similarities = F.log_softmax(similarity_matrix / self.temperature, dim=1)
        print('neg_similarities: ', neg_similarities)

        # Contrastive loss: sum of positive and negative log-sum-exp losses
        loss = -positive_similarities.mean() + neg_similarities.mean()
        print('loss: ', loss)


        return loss