# IPSVRG
The paper can be found here. Users should read the paper for problem descriptions and parameter definitions. There are three test problems: LASSO, logistic regression, and modified PCA (sum-of-nonconxex instance). The input data is a struct with 

      [data.A, data.b]

For LASSO and logistic regression, one can use online test data (LIBSVM); For modified PCA, the data is generated through 
      
      buildPCA.m

Call 

      InexactPrecdnTest.m 

with your test data to run our algorithm. Set parameters there, too. All parameters are consistent with notations on paper. 
