# README #

### What is this repository for? ###

* This is a Gaussian Process multi-class classification toolbox, in which Laplace Approximation is used for inference and maximising marginal likelihood is adapted to optimise the hyper-parameters of kernel functions.

* Version 1.01


### Who need this toolbox? ###

* This toolbox is design for those who want to solve multi-class classification and require the full predictive probabilities.


### How do I get set up? ###

(1) As this toolbox supports all kernels provided by GPML, you need to add GPML's toolbox (http://www.gaussianprocess.org/gpml/code/matlab/doc/) to the path. 

(2) run startup.m (GPML) to setup environment for GPML's toolbox.

(3) run demo.m (multi-class GPC)


### Contribution guidelines ###

This toolbox is mainly following GPML (Book: Gaussian Process for Machine Learning). We implement GP multi-class classification because it is not provided by GPML's toolbox.