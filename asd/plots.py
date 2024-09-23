import numpy as np
import matplotlib.pyplot as plt

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
        sensitivities = np.convolve(results.recalls, np.ones(window)/window, 'valid')
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
        plt.plot(epochs, train_losses, palette[0], label='train')
        plt.plot(epochs, test_losses, palette[1], label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # # Plot the metrics in the right subplot
        plt.subplot(1, n_plots, 2)
        plt.title(label="Performance metrics")
        plt.plot(epochs, accuracies, palette[0], label='accuracy')
        plt.plot(epochs, sensitivities, palette[1], label='sensitivity')
        plt.plot(epochs, precisions, palette[2], label='precision')
        plt.plot(epochs, f1s, palette[3], label='f1-measure')
        plt.plot(epochs, aucs, palette[4], label='aucs')

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')


        if update:
            plt.pause(1e-3)
        
        else:
            # save figure
            plt.show()


        