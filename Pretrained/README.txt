This folder contains all pretrained models.
Please see 'loadmodel.ipynb' for instruction to load the pretrained model parameters.

1. pretrained models of four cases: stable, unstable, midnoise and forcing
   "TR":   models trained with trust region newton cg method
   "ADAM": models trained with ADAM of two learning rates, 1e-3 and 1e-4
   
2. sensitivity analysis of FDNET
   "fdblock" :  models trained with trust region newton cg method for FD-Blocks
   "fdfilter":  models trained with trust region newton cg method for FD-Filters
