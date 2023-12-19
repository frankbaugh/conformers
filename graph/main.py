# from dihedral import run_epochs
# from dihedral_split import run_split_epochs
from limdihedral import run_epochs

run_epochs(epochs=50, learning_rate=1e-5, batch_size=4)
# run_split_epochs(epochs=50, learning_rate=8e-5)