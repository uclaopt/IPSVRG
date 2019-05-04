# IPSVRG
A light MATLAB package for acceleration of SVRG and Katyusha X by inexact preconditioning.

The paper can be found here. Users should read the paper for problem descriptions and parameter definitions. There are three test problems: LASSO, logistic regression, and modified PCA (sum-of-nonconxex instance). For LASSO and logistic regression, the input data is a struct of

      [data.A, data.b]
where data.A is the feature matrix and data.b is the label vector. One can download test data from e.g. LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/); For modified PCA, the input data is a struct of 
      
      [data.A, data.b, data.A_group, data.D_group]
which can be generated through calling
      
      buildPCA.m
The main function is

      InexactPrecdnTest.m 
including all parameter settings. Run our algorithm with the command

      InexactPrecdnTest(data).