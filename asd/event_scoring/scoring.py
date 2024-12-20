''' Scoring functions between a reference annotation (ground-truth) and hypotheses (e.g. ML output).
'''

import numpy as np
from asd.event_scoring.annotation import Annotation
import torch as th


class _Scoring:
    """" Base class for different scoring methods. The class provides the common
    attributes and computation of common scores based on these attributes.
    """
    fs: int
    numSamples: int

    refTrue: int
    tp: int
    fp: int

    sensitivity: float
    precision: float
    f1: float
    fpRate: float

    def computeScores(self):
        """ Compute performance metrics."""
        # Sensitivity
        if self.refTrue > 0:
            self.sensitivity = self.tp / self.refTrue
        else:
            self.sensitivity = np.nan  # no ref event

        # Precision
        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        else:
            self.precision = np.nan  # no hyp event

        # F1 Score
        if np.isnan(self.sensitivity) or np.isnan(self.precision):
            
            self.f1 = np.nan
        elif (self.sensitivity + self.precision) == 0:  # No overlap ref & hyp
            self.f1 = 0
        else:
            self.f1 = 2 * self.sensitivity * self.precision / (self.sensitivity + self.precision)

        # FP Rate
        self.fpRate = self.fp / (self.numSamples / self.fs / 3600 / 24) # FP per day


class EventScoring(_Scoring):
    """Calculates performance metrics on an event basis"""
    class Parameters:
        """Parameters for event scoring"""

        def __init__(self, toleranceStart: float = 60,
                     toleranceEnd: float = 60,
                     minOverlap: float = 0,
                     maxEventDuration: float = 100 * 60, # No limit
                     minDurationBetweenEvents: float = 90,
                     sampling_rate: float = 0.5
                     ):
            """Parameters for event scoring

            Args:
                toleranceStart (float): Allow some tolerance on the start of an event
                    without counting a false detection. Defaults to 30  # [seconds].
                toleranceEnd (float): Allow some tolerance on the end of an event
                    without counting a false detection. Defaults to 60  # [seconds].
                minOverlap (float): Minimum relative overlap between ref and hyp for
                    a detection. Defaults to 0 which corresponds to any overlap  # [relative].
                maxEventDuration (float): Automatically split events longer than a
                    given duration. Defaults to 5*60  # [seconds].
                minDurationBetweenEvents (float): Automatically merge events that are
                    separated by less than the given duration. Defaults to 90 # [seconds].
            """
            self.toleranceStart = toleranceStart
            self.toleranceEnd = toleranceEnd
            self.minOverlap = minOverlap
            self.maxEventDuration = maxEventDuration
            self.minDurationBetweenEvents = minDurationBetweenEvents
            self.fs = sampling_rate # Operate at a time precision of 256 Hz

    def __init__(self, ref_mask, hyp_mask, param: Parameters = Parameters(), fs: float = None):
        """Computes a scoring on an event basis.

        Args:
            ref (Annotation): Reference annotations (ground-truth)
            hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
            param(EventScoring.Parameters, optional):  Parameters for event scoring.
                Defaults to default values.
        """
        # Resample data
        self.fs = fs if fs is not None else param.fs
        self.ref = Annotation(ref_mask, self.fs)
        self.hyp = Annotation(hyp_mask, self.fs)
        
        # Apply sliding window
        self.hyp = EventScoring._applySlidingWindow(self.hyp, window_size=7, threshold=5)        

        self.ref = EventScoring._mergeNeighbouringEvents(self.ref, param.minDurationBetweenEvents)
        self.hyp = EventScoring._mergeNeighbouringEvents(self.hyp, param.minDurationBetweenEvents)
        # Split long events to param.maxEventDuration
        # self.ref = EventScoring._splitLongEvents(self.ref, param.maxEventDuration)
        # self.hyp = EventScoring._splitLongEvents(self.hyp, param.maxEventDuration)

        self.numSamples = len(self.ref.mask)
        self.refTrue = len(self.ref.events)

        # Count True detections
        self.tp = 0
        self.tpMask = np.zeros_like(self.ref.mask)
        extendedRef = EventScoring._extendEvents(self.ref, param.toleranceStart, param.toleranceEnd)
        for event in extendedRef.events:
            relativeOverlap = (np.sum(self.hyp.mask[round(event[0] * self.fs):round(event[1] * self.fs )])
                               ) / ((event[1] - event[0]) * self.fs)
            if relativeOverlap > param.minOverlap + 1e-6:
                self.tp += 1
                self.tpMask[round(event[0] * self.fs):round(event[1] * self.fs)] = 1

        # Count False detections
        self.fp = 0
        for event in self.hyp.events:
            if np.all(~self.tpMask[round(event[0] * self.fs):round(event[1] * self.fs)]):
                self.fp += 1
        self.computeScores()
        
    
    
    def _applySlidingWindow(annotation: Annotation, window_size: int = 7, threshold: int = 5) -> Annotation:
        """
        Apply a sliding window approach where at least 'threshold' out of 'window_size'
        segments need to be classified as a seizure to mark the entire window as a seizure.

        Args:
            annotation (Annotation): The input annotation object with a binary mask.
            window_size (int): The number of segments in the sliding window. Default is 7.
            threshold (int): The number of seizure segments needed within the window to classify
                            the whole window as a seizure. Default is 5 out of 7.

        Returns:
            Annotation: A new annotation object with the updated mask after applying the sliding window.
        """
        # Copy the original mask
        original_mask = annotation.mask.copy()
        new_mask = np.zeros_like(original_mask)  # Start with a new mask, all zeros
        
        num_segments = len(original_mask)
        
        # Slide the window across the entire mask
        for i in range(num_segments - window_size + 1):
            # Get the current window of size 'window_size'
            window = original_mask[i:i + window_size]
            
            # If 'threshold' or more segments in the window are seizures (1s), classify the window as a seizure
            if np.sum(window) >= threshold:
                # Set the entire window to 1 in the new mask
                new_mask[i:i + window_size] = 1
        
        # Create a new Annotation object with the updated mask
        return Annotation(new_mask, annotation.fs)


    def _splitLongEvents(events: Annotation, maxEventDuration: float) -> Annotation:
        """Split events longer than maxEventDuration in shorter events.
        Args:
            events (Annotation): Annotation object containing events to split
            maxEventDuration (float): maximum duration of an event [seconds]

        Returns:
            Annotation: Returns a new Annotation instance with all events split to
                a maximum duration of maxEventDuration.
        """

        shorterEvents = events.events.copy()

        for i, event in enumerate(shorterEvents):
            if event[1] - event[0] > maxEventDuration:
                shorterEvents[i] = (event[0], event[0] + maxEventDuration)
                shorterEvents.insert(i + 1, (event[0] + maxEventDuration, event[1]))
        
        return Annotation(shorterEvents, events.fs, len(events.mask))

    def _mergeNeighbouringEvents(events: Annotation, minDurationBetweenEvents: float) -> Annotation:
        """Merge events separated by less than longer than minDurationBetweenEvents.
        Args:
            events (Annotation): Annotation object containing events to split
            minDurationBetweenEvents (float): minimum duration between events [seconds]

        Returns:
            Annotation: Returns a new Annotation instance with events separated by less than
                minDurationBetweenEvents merged as one event.
        """

        mergedEvents = events.events.copy()

        i = 1
        while i < len(mergedEvents):
            event = mergedEvents[i]
            if (event[0] - mergedEvents[i - 1][1]) <= minDurationBetweenEvents:
                mergedEvents[i - 1] = (mergedEvents[i - 1][0], event[1])
                del mergedEvents[i]
                i -= 1
            i += 1

        return Annotation(mergedEvents, events.fs, len(events.mask))

    def _extendEvents(events: Annotation, before: float, after: float) -> Annotation:
        """Extend duration of all events in an Annotation object.

        Args:
            events (Annotation): Annotation object containing events to extend
            before (float): Time to extend before each event [seconds]
            after (float):  Time to extend after each event [seconds]

        Returns:
            Annotation: Returns a new Annotation instance with all events extended
        """

        extendedEvents = events.events.copy()
        fileDuration = len(events.mask) / events.fs

        for i, event in enumerate(extendedEvents):
            extendedEvents[i] = (max(0, event[0] - before), (min(fileDuration, event[1] + after)))

        return Annotation(extendedEvents, events.fs, len(events.mask))
    
    
    
def segments_to_events(lst):
    # Ensure the input is a list
    if not isinstance(lst, list):
        raise ValueError("Input must be a list")

    start_end_times = []
    in_sequence = False  # To track if we are in a sequence of 1's

    for i in range(len(lst)):
        if lst[i] == 1:
            if not in_sequence:
                # We've encountered the start of a new sequence
                start_index = i
                in_sequence = True
        else:
            if in_sequence:
                # We've reached the end of a sequence of 1's
                end_index = i - 1
                start_end_times.append((start_index, end_index))
                in_sequence = False

    # If the sequence ended at the last element, close it
    if in_sequence:
        end_index = len(lst) - 1
        start_end_times.append((start_index, end_index))

    # Convert indices to time in seconds
    time_segments = [(start * 4, (end + 1) * 4) for start, end in start_end_times]
    return lst,time_segments
