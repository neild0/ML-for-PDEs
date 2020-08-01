# Code for Submission to the 19th IEEE ICLMA, 2020.
  ## Python code of implementing FDNET, optimization method, experiments and figures of the main paper
# Submitted paper: 
  ## Finite Difference Neural Networks: Fast Prediction of Partial Differential Equations

# content:
  ## 1. Data generation code: DataGenerator/DataGene.py
     ### see instruction to generate data in dataloader.ipynb
     
  ## 2. Pretrained models: Pretrained/
     ### see instruction to load model parameters in loadmodel.ipynb
     ### four cases: stable, unstable, midnoise and force
     ### sensitivity analysis: fdblock and fdfilter
     
  ## 3. Optimization method: Optimizer/
     ### Trust Region Newton CG in TRCG.py
     
  ## 4. FDNET networks: Models/
     ### FDNET w/o forcing term: FDNET.py
     ### FDNET w/ forcing term: FDNET_FORCE.py
     
  ## 5. Figures: Plots/
     ### Figures 2 and 5: PlotGene_1.ipynb 
     ### Figures 6 and 7: PlotGene_2.ipynb
     ### Figure  3      : PlotGene_3.ipynb
     ### Figure  4      : PlotGene_4.ipynb
     
  ## 6. Train/Test:
     ### TR: Run_TR.ipynb
     ### ADAM: Run_ADAM.ipynb
  
  

