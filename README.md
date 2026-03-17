# catfda
categorical functional data analysis
Package: catfda

Type: Package

Title: categorical functional data analysis 

Version: 0.1.0

Author: Xiaoxia Champon, Chathura Jayalath

Maintainer: The package maintainer <xiachampon@gmail.com>

URL: http://github.com/XiaoxiaChampon/catfda

BugReports: http://github.com/XiaoxiaChampon/catfda/issues

Description: This package contains three functions:

             (1) Clustering: catfdcluster uses kmeans or DBSCAN on the multivariate 
	     
                 functional principal component scores extracted from the multivariate
		 
                 latent curves which induce the categorical functional data. The scores
		 
                 can be calculated using Ramsay's method or Happ's method.
		 
            (2) Hypothesis testing: aRLRTcfd uses  approximate restricted likelihood 
	    
                ratio test (aRLRT) to formally assess the relative importance of a category
		
                for the class membership. The test is conducted directly on the working 
		
                normalized response where the parameters are estimated with penalized quasi-likelihood.
		
            (3) Monitor clustering transition: catfdl incorporate the longitudinal nature of densely 
	    
                observed categorical valued functional data and monitors the clustering transitions
		
                over time through a single set of time varying functional principal component scores.

				
License: 

Encoding: UTF-8

LazyData: true

Imports:

    fda,
    
    refund,
    
    mgcv,
    
    funData,
    
    MFPCA,
    
    dbscan,
    
    fossil,
    
    NbClust,
    
    ggplot2
    
    
    To use the functions in the package:
    
  1) library(devtools)
  
     install_github("XiaoxiaChampon/catfda")
     
  2) library("catfda")

     catfdclust=catfdcluster(matrix(sample(c(0,1,2),100*250,replace=TRUE),nrow=100,ncol=250),seq(0,1,length=250),25,3,3,0.9,4,5,2,"happ","two")
     
   3) sample code is in the R folder: cluster_sample.R

## Development Setup

**Quick Start:** For a fast setup guide to connect Visual Studio Code, see [QUICKSTART.md](QUICKSTART.md)

For detailed instructions on how to set up and use this package with Visual Studio Code, see [VISUAL_STUDIO_SETUP.md](VISUAL_STUDIO_SETUP.md).

