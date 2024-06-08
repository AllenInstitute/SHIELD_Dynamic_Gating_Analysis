# Batch analysis of Dynamic Gating Sessions
This code was used to generate intermediate data structures from the primary NWBs to facilitate further analysis. 


# Environment Setup
The recommended environment to use the data is `python>=3.9`. There are two main dependencies that must be installed as follows: 

Fork of AllenSDK to handle Dynamic Gating sessions:
`pip install git+https://github.com/arjunsridhar12345/AllenSDK`.

Helpful package with utilities to analyze Brain Observatory data:

`git clone https://github.com/AllenInstitute/brain_observatory_utilities.git
cd brain_observatory_utilities
pip install -e .
`
