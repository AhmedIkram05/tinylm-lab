"""The one Okabe-Ito palette shared by the training plot and the sweep tooling.

Colour-blind safe, from the Okabe & Ito palette. Kept in its own module so the
library plot (`train.py`) and the tooling plots import it in the same direction
instead of one reaching into the other.
"""

TRAIN_BLUE = "#0072B2"
VAL_VERMILION = "#D55E00"
