import seizure_data_processing as sdp




class Dataset:
    
    
    def __init__(self, path=None, channels=None):
        # assert path is not None, "No dataset provided"
        # assert channels is not None, "No channels provided"
        
        # Load a single file
        file = (
                r"./data/chb23_06.edf"
        )
        channels = "FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8;T8-P8".split(";")
        self.eeg_file = sdp.EEG(file, channels=channels)
    
    def data(self):
        return self.eeg_file.data
    
    def labels(self):
        return self.eeg_file.get_labels()
    
    def annotations(self):
        return self.eeg_file.annotations
    
    def plot(self):
        self.eeg_file.plot()