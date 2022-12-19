# Title: Regularization-Penalty-Optimization-for-Addressing-Data-Quality-Variance-in-Ood-Algorithms
# The py. files for this project are designed based on the github below:

1) https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
2) https://github.com/kakaobrain/irm-empirical-study/tree/master/colored_mnist

## Brief Deiscription for code implementation

1) Most of the progress of making enviornments is designed following the code that uploaded on the url above(especially train dataset vs test dataset). However additional progress is added to making enviornments based on train dataset. Differently with the original code where the degree of switching labels are same for both enviornments (25%), I modified the degree to 0% and 50% for each enviornment to make the difference of data quality clearly.

# description for py.file
1) Base_experiment.py : Experiment without considering the OOD generalization
2) IRM_experiment.py : Experiment with IRM algorithm which is the base model for this paper
3) IRM_ver1_experiment.py : Experiment with modified IRM algorithm where the values of domain-level weights are estimated for each of the domain
4) IRM_rev_experiment.py : Experiment with proposed model (paper) where both the domain-level and sample level regularizers are added to the original IRM algorithm.
