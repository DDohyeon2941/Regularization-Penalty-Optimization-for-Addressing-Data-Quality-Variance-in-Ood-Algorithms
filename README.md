# Title: Regularization-Penalty-Optimization-for-Addressing-Data-Quality-Variance-in-Ood-Algorithms
# The py. files for this project are designed based on the github below:

1) https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
2) https://github.com/kakaobrain/irm-empirical-study/tree/master/colored_mnist

## Brief Description for code implementation

1) Most of the progress of making environments is designed following the code that uploaded on the url above(especially train dataset vs test dataset). However additional progress is added to making environments based on train dataset. Differently with the original code where the degree of switching labels are same for both environments (25%), I modified the degree to 0% and 50% for each enviornment to make the difference of data quality clearly.

2) Calculation for the major sub-elements for the proposed methods[the progress are applied to the IRM_ver1_experiment.py and IRM_rev_experiment.py]
The characteristics of progress of calculating one of the sub-elements representing noise-level (error) of the dataset is summarized bellow:

- I replace the loss function from MSE to NLL by considering that only binary classification problem is treated for this project.
- I use the trained models for previous restart as the 'extra model' which is described on the paper (for every restart, model and enviornments are defined newly) 


## description for py.file
1) Base_experiment.py : Experiment without considering the OOD generalization
2) IRM_experiment.py : Experiment with IRM algorithm which is the base model for this paper
3) IRM_ver1_experiment.py : Experiment with modified IRM algorithm where the values of domain-level weights are estimated for each of the domain
4) IRM_rev_experiment.py : Experiment with proposed model (paper) where both the domain-level and sample level regularizers are added to the original IRM algorithm.
