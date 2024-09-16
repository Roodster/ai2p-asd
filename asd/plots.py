import numpy as np
import matplotlib.pyplot as plt

class Plots:
    
    
    def __init__(self):
        pass
    
    
    def plot(self, results, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves

        window = 100
        
        if len(results.epochs) < 2: return
        epochs = np.convolve(results.epochs, np.ones(window)/window, 'valid')
        train_losses = np.convolve(results.train_losses, np.ones(window)/window, 'valid')
        test_losses = np.convolve(results.test_losses, np.ones(window)/window, 'valid')

        tps = np.convolve(results.tps, np.ones(window)/window, 'valid')
        fps = np.convolve(results.fps, np.ones(window)/window, 'valid')
        tns = np.convolve(results.tns, np.ones(window)/window, 'valid')
        fns = np.convolve(results.fns, np.ones(window)/window, 'valid')
              
        sensitivities = (tps / (tps + fns))
        precisions = (tps / (tps + fps))
        f1s = 2 * (precisions * sensitivities) / (precisions + sensitivities)

        # Determine x-axis based on samples or episodes
        # Create plot
        colors = ['r', 'g', 'b']


        n_plots = 2
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        # Plot the losses in the left subplot
        plt.subplot(1, n_plots, 1)
        plt.title(label="Train/Test Loss")
        plt.plot(epochs, train_losses, colors[0], label='train')
        plt.plot(epochs, test_losses, colors[2], label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # # Plot the metrics in the right subplot
        plt.subplot(1, n_plots, 2)
        plt.title(label="Performance metrics")
        plt.plot(epochs, sensitivities, colors[0], label='sensitivity')
        plt.plot(epochs, precisions, colors[1], label='precision')
        plt.plot(epochs, f1s, colors[2], label='f1-measure')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')


        if update:
            plt.pause(1e-3)
        
        else:
            # save figure
            plt.show()


        