import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mc
import colorsys
import numpy as np

from .annotation import Annotation
from .scoring import EventScoring

def plotEventScoring(ref: Annotation, hyp: Annotation,
                     param: EventScoring.Parameters = EventScoring.Parameters(),
                     showLegend: bool = True, ax: Axes = None) -> plt.figure:
    """Build an overview plot showing the outcome of event scoring.

    If an axes is provided, plots on that axes, else creates a new figure.

    Args:
        ref (Annotation): Reference annotations (ground - truth)
        hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
        param(EventScoring.Parameters, optional):  Parameters for event scoring.
            Defaults to default values.
        showLegend (bool): Whether to show the legend. Default True.
        ax (Axes): If provided figure is plotted on that axes. Else a new figure is created. Default None (new figure).

    Returns:
        plt.figure: Output matplotlib figure
    """
    score = EventScoring(ref.mask, hyp.mask, param)
    time = np.arange(len(ref.mask)) / ref.fs

    if ax is None:
        plt.figure(figsize=(16, 3))
        ax = plt.axes()

    # Plot Labels
    ax.plot(time, ref.mask * 0.4 + 0.6, 'k')
    ax.plot(time, hyp.mask * 0.4 + 0.1, 'k')
    
    # Plot splitting of events
    for event in ref.events:
        _plotSplitLongEvents(event, param.maxEventDuration, [0.6, 1])
    for event in hyp.events:
        _plotSplitLongEvents(event, param.maxEventDuration, [0.1, 0.5])

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
        _plotEvent([event[0], event[1] - (1 / ref.fs)], [1, 1], color, ax,
                   [max(0, event[0] - param.toleranceStart), min(time[-1], event[1] + param.toleranceEnd - (1 / ref.fs))])

    # Plot HYP TP & FP
    for event in score.hyp.events:
        # FP
        if np.all(~score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red', ax)
        # TP
        elif np.all(score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            ax.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                    color='tab:green', linewidth=5, solid_capstyle='butt', linestyle='solid')
        # Mix TP, FP
        else:
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red', ax, zorder=1.7)
            ax.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                    color='tab:green', linewidth=5, solid_capstyle='butt', linestyle=(0, (2, 2)))

    # Text
    plt.title('Event Scoring')

    ax.set_yticks([0.3, 0.8], ['HYP', 'REF'])
    _scale_time_xaxis(ax)

    if showLegend:
        _buildLegend(lineTp, lineFn, lineFp, score, ax)


    plt.show()  # Ensure the plot is shown in Kaggle

    return plt.gcf()

def plotIndividualEvents(ref: Annotation, hyp: Annotation,
                         param: EventScoring.Parameters = EventScoring.Parameters()) -> plt.figure:
    """Plot each individual event in event scoring.
    Events are organized in a grid with the evennts centered in 5 minute windows.

    Args:
        ref (Annotation): Reference annotations (ground - truth)
        hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
        param(EventScoring.Parameters, optional):  Parameters for event scoring.
            Defaults to default values.

    Returns:
        plt.figure: Output matplotlib figure
    """
    score = EventScoring(ref.mask, hyp.mask, param)

    # Get list of windows to plot (windows are 5 minutes long centered around events)
    duration = 5 * 60
    listofWindows = list()
    plottedMask = np.zeros_like(score.ref.mask)
    for i, event in enumerate(score.ref.events + score.hyp.events):
        center = event[0] + (event[1] - event[0]) / 2
        window = (max(0, center - duration / 2), min(len(plottedMask) / score.fs, center + duration / 2))

        if not np.all(plottedMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            plottedMask[round(window[0] * score.fs):round(window[1] * score.fs)] = 1
            listofWindows.append(window)

    # Plot windows in a grid configuration
    NCOL = 3
    nrow = int(np.ceil(len(listofWindows) / NCOL))
    plt.figure(figsize=(16, nrow * 2))
    for i, window in enumerate(listofWindows):
        ax = plt.subplot(nrow, NCOL, i + 1)
        plotEventScoring(ref, hyp, showLegend=False, ax=ax)
        ax.set_xlim(window)
        plt.title('Event {}'.format(i))
    plt.tight_layout()
    
    plt.show()  # Ensure the plot is shown in Kaggle
    return plt.gcf()


def _scale_time_xaxis(ax: Axes):
    """Scale x axis of a figure where initial values are in seconds.

    The function leaves the xaxis as is if the number of seconds to display is < 5 * 60
    If it is larger than 5 minutes, xaxis is formatted as m:s
    If it is larger than 5 hours, xaxis is formatted as h:m:s

    Args:
        ax (Axes): axis to handle
    """

    def s2m(x, _):
        return f'{int(x / 60)}:{int(x%60)}'

    def s2h(x, _):
        return f'{int(x / 3600)}:{int((x / 60)%60)}:{int(x%60)}'

    maxTime = ax.get_xlim()[1]
    if maxTime > 5 * 60 * 60:
        ax.xaxis.set_major_formatter(s2h)
        ax.set_xlabel('time [h:m:s]')
    elif maxTime > 5 * 60:
        ax.xaxis.set_major_formatter(s2m)
        ax.set_xlabel('time [m:s]')
    else:
        ax.set_xlabel('time [s]')


def _buildLegend(lineTp, lineFn, lineFp, score, ax):
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


def _plotEvent(x, y, color, ax, bckg=None, zorder=1.8):
    if bckg is None:
        bckg = x
    ax.axvspan(bckg[0], bckg[1], color=adjust_lightness(color, 0.2), zorder=zorder)
    if x[1] - x[0] > 0:
        ax.plot(x, y, color=color, linewidth=5, solid_capstyle='butt')
    else:
        ax.scatter(x[0], y[0], color=color)


def _plotSplitLongEvents(event, maxEventDuration, y):
    """ Visualize split of long events """
    t = event[0] + maxEventDuration
    while t < event[1]:
        plt.plot([t, t], y, '--k', zorder=1.9)
        t += maxEventDuration


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])