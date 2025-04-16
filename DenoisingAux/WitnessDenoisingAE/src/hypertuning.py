import sys
import os
sys.path.append(os.path.relpath("../src"))
from training_CUPID import tune_hyper
tune_hyper("/feynman/work/projets/cupid/scratch/mpageot/data/trainings/opt_AutoencoderLSTM_RUN9_813_channel19_4/",n_trials=20)
