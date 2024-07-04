# SHIELD_Dynamic_Gating_Analysis
Code used in Bennett et al., SHIELD: Skull-shaped hemispheric implants enabling large-scale electrophysiology datasets in the
mouse brain, Neuron (2024), https://doi.org/10.1016/j.neuron.2024.06.015. 

This code is published on zenodo: [![DOI](https://zenodo.org/badge/811016180.svg)](https://zenodo.org/doi/10.5281/zenodo.11494005)

# Environment Setup
The recommended environment is `python>=3.9`. There is one main dependency that must be installed as follows: 

Fork of AllenSDK to handle Dynamic Gating sessions:

`pip install git+https://github.com/arjunsridhar12345/AllenSDK`.

# Using the data
The primary data used for analysis in this repo has been published on DANDI [here](https://dandiarchive.org/dandiset/001051).
You can find tutorials on how to analyze these NWB files in the tutorials folder. 

These data are highly similar to the data released for the Allen Observatory Visual Behavior Neuropixels project. Potential users are encouraged to consult the resources for that project [here](https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels) for more detailed information about the NWB organization. 
