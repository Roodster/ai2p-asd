import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plots:
    
    
    def __init__(self):
        pass
    
    
    def plot(self, results, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves

        window = max(int(len(results.epochs) // 10), 1)
        
        if len(results.epochs) < 2: return
        epochs = np.convolve(results.epochs, np.ones(window)/window, 'valid')
        train_losses = np.convolve(results.train_losses, np.ones(window)/window, 'valid')
        test_losses = np.convolve(results.test_losses, np.ones(window)/window, 'valid')

        precisions = np.convolve(results.precisions, np.ones(window)/window, 'valid')
        sensitivities = np.convolve(results.sensitivities, np.ones(window)/window, 'valid')
        accuracies = np.convolve(results.accuracies, np.ones(window)/window, 'valid')
        f1s = np.convolve(results.f1s, np.ones(window)/window, 'valid')
        aucs = np.convolve(results.aucs, np.ones(window)/window, 'valid')
        
        # Determine x-axis based on samples or episodes
        # Create plot


        n_plots = 2
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        # Choose a color palette (e.g., "colorblind")
        palette = sns.color_palette("colorblind", n_colors=5)
        plt.clf()                
        # Plot the losses in the left subplot
        plt.subplot(1, n_plots, 1)
        plt.title(label="Train/Test Loss")
        plt.plot(epochs, train_losses, color=palette[0], label='train')
        plt.plot(epochs, test_losses, color=palette[1], label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # # Plot the metrics in the right subplot
        plt.subplot(1, n_plots, 2)
        plt.title(label="Performance metrics")
        plt.plot(epochs, accuracies, color=palette[0], label='accuracy')
        plt.plot(epochs, sensitivities, color=palette[1], label='sensitivity')
        plt.plot(epochs, precisions, color=palette[2], label='precision')
        plt.plot(epochs, f1s, color=palette[3], label='f1-measure')
        plt.plot(epochs, aucs, color=palette[4], label='aucs')

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')


        if update:
            plt.pause(1e-3)
        
        else:
            # save figure
            plt.show()

    def plot_attention_maps(self, attention_scores, layer_idx=0, head_idx=0):
        """
        Visualize the attention scores for a specific layer and attention head.

        Args:
            attention_scores (list of tensors): List of attention scores for each layer.
            layer_idx (int): Layer index to visualize.
            head_idx (int): Attention head index to visualize.
        """
        # Select the attention score of the specified layer and head
        attn = attention_scores[layer_idx][0, head_idx].detach().cpu().numpy()

        # Plot the attention map
        plt.figure(figsize=(8, 8))
        plt.imshow(attn, cmap='viridis')
        plt.title(f"Attention Map - Layer {layer_idx}, Head {head_idx}")
        plt.colorbar()
        plt.show()
        
    def plot_aggregated_attention(self, attention_scores, layer_idx=0, mode='mean'):
            """
            Visualize the aggregated attention scores for a specific layer, independent of attention heads.

            Args:
                attention_scores (list of tensors): List of attention scores for each layer.
                layer_idx (int): Layer index to visualize.
                mode (str): Aggregation mode for heads. Options are 'mean' or 'sum'.
            """
            # Aggregate attention scores across heads
            if mode == 'mean':
                aggregated_attention = attention_scores[layer_idx].mean(dim=1)  # Average over heads
            elif mode == 'sum':
                aggregated_attention = attention_scores[layer_idx].sum(dim=1)  # Sum over heads
            else:
                raise ValueError("Invalid mode. Choose 'mean' or 'sum'.")

            # Extract the attention map for the first input sequence
            attn = aggregated_attention[0].detach().cpu().numpy()

            # Plot the aggregated attention map
            plt.figure(figsize=(8, 8))
            plt.imshow(attn, cmap='viridis')
            plt.title(f"Aggregated Attention Map - Layer {layer_idx} ({mode})")
            plt.colorbar()
            plt.show()