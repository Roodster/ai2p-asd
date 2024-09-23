import torch as th
import torch.nn.functional as F

class OHELabelTransformer(th.nn.Module):
    """Class to transform class labels using one-hot encoding."""

    def __init__(self, num_classes=2):
        """
        Initializes the label transformer.

        Args:
            num_classes: The total number of classes.
        """
        super(OHELabelTransformer, self).__init__()
        self.num_classes = num_classes

    def forward(self, y):
        """
        Transforms the input labels using one-hot encoding.

        Args:
            x: A tensor containing the input labels.

        Returns:
            A tensor containing the one-hot encoded labels.
        """
        # Convert labels to one-hot encoding
        one_hot = F.one_hot(y.to(th.int64), num_classes=self.num_classes)

        return one_hot