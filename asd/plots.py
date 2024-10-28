import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mc
import colorsys
import numpy as np
from asd.event_scoring.annotation import Annotation
from asd.event_scoring.scoring import EventScoring


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
            
            

class EventPlots:
    
    def __init__(self):
        pass
    
    def plot(self, results, update=True):
        """Plots logged training results including performance metrics, FP/day, and losses.
           Use 'update=True' if the plot is continuously updated
           or use 'update=False' if this is the final call (to avoid double plotting)."""
        # Smooth curves
        window = max(int(len(results.epochs) // 10), 1)
        
        if len(results.epochs) < 2: return
        
        # Smooth the data for plotting
        epochs = np.convolve(results.epochs, np.ones(window)/window, 'valid')
        train_losses = np.convolve(results.train_losses, np.ones(window)/window, 'valid')
        test_losses = np.convolve(results.test_losses, np.ones(window)/window, 'valid')
        precisions = np.convolve(results.precisions, np.ones(window)/window, 'valid')
        sensitivities = np.convolve(results.sensitivities, np.ones(window)/window, 'valid')
        f1s = np.convolve(results.f1s, np.ones(window)/window, 'valid')
        fp_rates = np.convolve(results.fp_rates, np.ones(window)/window, 'valid')
        
        # Set up a plot with three subplots
        n_plots = 3
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        
        # Choose a color palette
        palette = sns.color_palette("colorblind", n_colors=3)
        
        plt.clf()

        # Plot the losses in the first subplot
        plt.subplot(1, n_plots, 1)
        plt.title("Train/Test Loss")
        plt.plot(epochs, train_losses, color=palette[0], label='train')
        plt.plot(epochs, test_losses, color=palette[1], label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Plot precision, sensitivity, and F1 score in the second subplot
        plt.subplot(1, n_plots, 2)
        plt.title("Performance Metrics")
        plt.plot(epochs, precisions, color=palette[0], label='precision')
        plt.plot(epochs, sensitivities, color=palette[1], label='sensitivity')
        plt.plot(epochs, f1s, color=palette[2], label='f1-measure')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')

        # Plot FP/day in the third subplot
        plt.subplot(1, n_plots, 3)
        plt.title("False Positive Rate (FP/day)")
        plt.plot(epochs, fp_rates, color='red', label='FP/day', linestyle='--')  # FP/day with dashed line
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('FP/day')

        # Display the plot
        if update:
            plt.pause(1e-3)
        else:
            # Save or show the final figure
            plt.show()
            


    def plotEventScoring(self, ref: Annotation, hyp: Annotation,
                         param: EventScoring.Parameters = EventScoring.Parameters(),
                         showLegend: bool = True, ax: Axes = None) -> plt.figure:
        """Build an overview plot showing the outcome of event scoring.
           If an axes is provided, plots on that axes, else creates a new figure."""
        score = EventScoring(ref.mask, hyp.mask, param, fs = ref.fs)
        time = np.arange(len(ref.mask)) / ref.fs / 2
        if ax is None:
            plt.figure(figsize=(16, 3))
            ax = plt.axes()

        # Plot Labels
        ax.plot(time, ref.mask * 0.4 + 0.6, 'k')
        ax.plot(time, hyp.mask * 0.4 + 0.1, 'k')
        
        # Initialize lines for legend
        lineTp, = ax.plot([], [], color='tab:green', linewidth=5)
        lineFn, = ax.plot([], [], color='tab:purple', linewidth=5)
        lineFp, = ax.plot([], [], color='tab:red', linewidth=5)

        # Plot REF TP & FN
        for event in score.ref.events:
            # TP
            if np.any(score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
                color = 'tab:green'
            else:
                color = 'tab:purple'
            self._plotEvent([event[0], event[1] - (1 / ref.fs)], [1, 1], color, ax,
                            [max(0, event[0] - param.toleranceStart), min(time[-1], event[1] + param.toleranceEnd - (1 / ref.fs))])

        # Plot HYP TP & FP
        for event in score.hyp.events:
            # FP
            if np.all(~score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
                self._plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red', ax)
            # TP
            elif np.all(score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
                ax.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                        color='tab:green', linewidth=5, solid_capstyle='butt', linestyle='solid')
            # Mix TP, FP
            else:
                self._plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red', ax, zorder=1.7)
                ax.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                        color='tab:green', linewidth=5, solid_capstyle='butt', linestyle=(0, (2, 2)))

        # Text
        plt.title('Event Scoring')

        ax.set_yticks([0.3, 0.8], ['HYP', 'REF'])
        self._scale_time_xaxis(ax)

        if showLegend:
            self._buildLegend(lineTp, lineFn, lineFp, score, ax)

        plt.show()

        return plt.gcf()
    
    def plotIndividualEvents(self, ref: Annotation, hyp: Annotation,
                             param: EventScoring.Parameters = EventScoring.Parameters()) -> plt.figure:
        """Plot each individual event in event scoring.
        Events are organized in a grid with the events centered in 5 minute windows.
    
        Args:
            ref (Annotation): Reference annotations (ground-truth)
            hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
            param(EventScoring.Parameters, optional):  Parameters for event scoring.
                Defaults to default values.
    
        Returns:
            plt.figure: Output matplotlib figure
        """
        score = EventScoring(ref.mask, hyp.mask, param, ref.fs)
    
        # Get list of windows to plot (windows are 5 minutes long centered around events)
        duration = 30 * 60  # 5-minute window
        listofWindows = []
        plottedMask = np.zeros_like(score.ref.mask)
    
        for event in score.ref.events +  score.hyp.events :
            # Center the window around the event
            center = event[0] + (event[1] - event[0]) / 2
            window_start = max(0, center - duration / 2) / 2
            window_end = min(len(plottedMask) / score.fs, center + duration / 2) / 2
            window = (window_start, window_end)
    
            # Ensure this event hasn't been plotted before
            if not np.all(plottedMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
                plottedMask[round(window[0] * score.fs):round(window[1] * score.fs)] = 1
                listofWindows.append(window)
    
        # Plot windows in a grid configuration
        NCOL = 2  # Change this to make the grid wider or narrower
        nrow = int(np.ceil(len(listofWindows) / NCOL))
        plt.figure(figsize=(16, nrow * 2))
    
        for i, window in enumerate(listofWindows):
            ann1 = Annotation(ref.mask[(int(window[0] * ref.fs)):(int(window[1] * ref.fs))], fs = ref.fs)
            ann2 = Annotation(hyp.mask[int(window[0] * hyp.fs):int(window[1] * hyp.fs)], fs = hyp.fs)
            # print("Hyp: ", hyp.mask[int(window[0] * hyp.fs):int(window[1] * hyp.fs)])
            # print("Ref: ", ref.mask[int(window[0] * ref.fs):int(window[1] * ref.fs)])
            
            self.plotEventScoring(ann1, ann2)
            # print(f"Window {i}: {window}")  # Debug print to verify window ranges
        
        return plt.gcf()


    def _scale_time_xaxis(self, ax: Axes):
        """Scale x axis of a figure where initial values are in seconds."""
        def s2m(x, _):
            return f'{int(x / 60)}:{int(x % 60)}'

        def s2h(x, _):
            return f'{int(x / 3600)}:{int((x / 60) % 60)}:{int(x % 60)}'

        maxTime = ax.get_xlim()[1]
        if maxTime > 5 * 60 * 60:
            ax.xaxis.set_major_formatter(s2h)
            ax.set_xlabel('time [h:m:s]')
        elif maxTime > 5 * 60:
            ax.xaxis.set_major_formatter(s2m)
            ax.set_xlabel('time [m:s]')
        else:
            ax.set_xlabel('time [s]')

    def _buildLegend(self, lineTp, lineFn, lineFp, score, ax):
        """Build legend and adjust spacing for scoring text"""
        ax.legend([lineTp, lineFn, lineFp],
                  ['TP: {}'.format(np.sum(score.tp)),
                   'FN: {}'.format(np.sum(score.refTrue - score.tp)),
                   'FP: {}'.format(np.sum(score.fp))], loc=(1.02, 0.65))

        textstr = "• Sensitivity: {:.2f}\n".format(score.sensitivity)
        textstr += "• Precision  : {:.2f}\n".format(score.precision)
        textstr += "• F1 - score   : {:.2f}".format(score.f1)
        ax.text(1.02, 0.05, textstr, fontsize=12, transform=ax.transAxes)

        # Adjust spacing
        ax.margins(x=0)  # No margin on X data
        plt.tight_layout()
        plt.subplots_adjust(right=0.86)  # Allow space for scoring text


    def _plotEvent(self, x, y, color, ax, bckg=None, zorder=1.8):
        if bckg is None:
            bckg = x
        ax.axvspan(bckg[0], bckg[1], color=self.adjust_lightness(color, 0.2), zorder=zorder)
        if x[1] - x[0] > 0:
            ax.plot(x, y, color=color, linewidth=5, solid_capstyle='butt')
        else:
            ax.scatter(x[0], y[0], color=color)

    def _plotSplitLongEvents(self, event, maxEventDuration, y):
        """ Visualize split of long events """
        t = event[0] + maxEventDuration
        while t < event[1]:
            plt.plot([t, t], y, '--k', zorder=1.9)
            t += maxEventDuration

    def adjust_lightness(self, color, amount=0.5):
        try:
            c = mc.cnames[color]
        except KeyError:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])