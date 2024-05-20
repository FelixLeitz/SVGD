# Stein variational gradient descent for Bayesian Neural Networks
Contains the code used for my project work **Applicability of SVGD for high-dimensional problems using Neural Networks** and the results discussed in the corresponding paper. The work was supervised by the Department of Stochastic Simulation and Safety Research for Hydrosystems (LS3) at the university of Stuttgart.

## Content
To properly understand everything, basic knowledge of neural networks and sampling methods is required.

The main goal of the presented work is to assess wether it is feasible to use SVGD for uncertainty quantification in neural networks. The combination of SVGD and NNs poses an interesting task, as Bayesian inference on neural networks is a non-trivial problem especially with large networks. Furthermore, with SVGD being a gradient-based method the straightforward gradient access in a neural network is helpful.

## Sources

'''
@misc{SVGDintro,
  title	={Stein Variational Gradient Descent: A general purpose Bayesian inference algorithm},
  author={Qiang Liu and Dilin Wang},
  year	={2016},
  institution={Dartmouth College}
}
'''
