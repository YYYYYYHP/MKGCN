# MKGCN
**MKGCN: a novel method for microbe-drug associations prediction via multiple kernel fusion on graph neural network.**

![image](http://yhpjc.vip/download/MKGCN/flowchart.png)

# Dataset
1)MDAD: including 5505 clinically or experimentally verified microbe-drug associations, between 1388 drugs and 174 microbes;

2)aBiofilm: including resource of anti-biofilm agents and their potential implications in antibiotic drug resistance;

3)DrugVirust: including the activity and development of related compounds of a variety of human viruses;

# Data description
* adj: interaction pairs between microbes and drugs.
* drugs: IDs and names for drugs.
* microbes/viruses: IDs and names for microbes/viruses.
* drugfeatures: pre-processing feature matrix for drugs.
* microbefeatures: pre-processing feature matrix for microbes.
* drugsimilarity: integrated drug similarity matrix.
* microbesimilarity: integrated microbe similarity matrix.

# Requirements
* Python 3.7
* Pytorch
* PyTorch Geometric
* numpy
* scipy
