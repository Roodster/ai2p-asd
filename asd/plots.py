import numpy as np
import matplotlib.pyplot as plt

class Plots:
    
    
    def __init__(self):
        pass
    
    
    def plot(self, results, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves


        print("\n", results.get())
        window = max(len(results.epochs) / 50 , 1)
        
        if len(results.epochs) < window + 1: return
        epochs = np.convolve(results.epochs, np.ones(window)/window, 'valid')
        train_losses = np.convolve(results.train_losses, np.ones(window)/window, 'valid')
        test_losses = np.convolve(results.test_losses, np.ones(window)/window, 'valid')

        print('result tps: ', results.tps)
        print('result fps: ', results.fps)
        print('result fns: ', results.fns) 

        tps = np.convolve(results.tps, np.ones(window)/window, 'valid')
        fps = np.convolve(results.fps, np.ones(window)/window, 'valid')
        tns = np.convolve(results.tns, np.ones(window)/window, 'valid')
        fns = np.convolve(results.fns, np.ones(window)/window, 'valid')
        
        
        print('tps: ', tps)
        print('fps: ', fps)
        print('fns: ', fns)        
        sensitivities = (tps / (tps + fns))
        print('sens: ', sensitivities)
        precisions = (tps / (tps + fps))
        print('prec: ', precisions)
        f1s = 2 * (precisions * sensitivities) / (precisions + sensitivities)
        print('f1: ', f1s)

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
        # ax.plot(env_steps, lengths, colors[0])
        # ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
        # ax.set_ylabel('episode length')
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


        