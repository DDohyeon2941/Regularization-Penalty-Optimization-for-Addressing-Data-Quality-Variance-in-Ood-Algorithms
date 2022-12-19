# Title: Regularization-Penalty-Optimization-for-Addressing-Data-Quality-Variance-in-Ood-Algorithms
# The py. files for this project are designed based on the github below:

1) https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
2) https://github.com/kakaobrain/irm-empirical-study/tree/master/colored_mnist


# description py.file
1) Base_experiment.py : Experiment without considering the OOD generalization
2) IRM_experiment.py : Experiment with IRM algorithm which is the base model for this paper
3) IRM_ver1_experiment.py : Experiment with modified IRM algorithm where the values of domain-level weights are estimated for each of the domain
4) IRM_rev_experiment.py : Experiment with proposed model (paper) where both the domain-level and sample level regularizers are added to the original IRM algorithm.
