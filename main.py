from asd.make_dataset import get_pi_dataloader
from asd.args import Args





if __name__ == "__main__":
    
    channels = "FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8;T8-P8".split(";")

    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    # Load dataset
    dataloader = get_pi_dataloader(path='./data/dataset/data/', channels=channels, batch_size=128, shuffle=False)
    
    
    # Train and evaluate
    
    