# IFENN-with-PICNNs-for-phase-field-propagation

This repository contains the code used in this article: "Integrated Finite Element Neural Network (IFENN) for phase-field fracture with minimal input and generalized geometry-load handling", (Computer Methods in Applied Mechanics and Engineering, 2025).

The code can be run in two modes:
a) "train_mode": utilized to train the PICNN from scratch
b) "pred_mode": used to conduct a pure-FEM analysis (IFENN_flag == "off") or IFENN analysis (IFENN_flag == "on").

IFENN can be activated either:
a) based on the maximum value of phase-field ("IFENN_switch_criterion == dmax_based"), or
b) at a predefined load increment ("IFENN_switch_criterion == inc_based").

By default, the code is configured to reproduce the Single Notch Tension (SNT1) simulation reported in the paper. Additionally, we provide the necessary files ("model.msh" and "index_pixel_model.mat") to reproduce the other benchmark problems as well.

For any queries, please reach out to Panos Pantidis (pp262@nyu.edu), Lampros Svolos (Lampros.Svolos@uvm.edu) or Mostafa E. Mobasher (mostafa.mobasher@nyu.edu).
