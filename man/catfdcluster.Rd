\name{catfdcluster}
\alias{catfdcluster}



\title{cluster categorical functional data}

\description{
This function clusters categorical functional data, where each series are categorical values observed on a dense interval. It uses kmeans or DBSCAN to cluster on the multivariate functional principal component scores extracted from the multivariate latent cruves, which induce the categorical functional data. Ramsay method concatenates the multivariate latent curves and the final scores are the linear combinations of the scores from each component. Happ method uses univariate expansion and can consider the weight of each component or different domains.

}
\usage{
catfdcluster(catfd,argval,splines1D,M,knnum,pct,minPts,max.nc,min.nc,method,numdim)
}


\arguments{
  \itemize{
   \item{catfd}{: matrix n rows is the number of individuals, T columns is the number of time total time points. Each row represents a categorical series observed on a dense interval at each point t. }

  \item {argval} {: vector length T. The observational time interval.}
  \item {splines1D} {: scalar the number of univariate basis needed. Usually 25.}
  \item{M}{: scalar The number of principa components. Usually 3.}
  \item{knnum}{: scalar The number of neighbours to calculate the distance in DBSCAN}
  \item{pct}{: scalar between 0 and 1: The percentage of the distance used to determin the epsilon in DBSCAN. }
  \item{minPts}{: scalar The minimum number of points needed to define a core point in DBSCAN}
  \item{max.nc}{: scalar maximum number of clusters wanted in Kmeans}
  \item{min.nc}{: scalar minimum number of clusters wanted in Kmeans}
  \item{method}{: "ramsay" or "happ" Ramsay concatenates the multivariate latent curves and the final scores are the linear combinations of the scores from each component. Happ can account the weight among different components and handle different domains.}
  \item{numdim}{: "all" or "two". "all" uses all the dimensions of the mfpca scores that explain at least 95 percent of the variance, "two" uses only the first two dimensions}
  }

}


\details{
See Clustering of categorical valued functional data with application to social media
}
\value{
  \itemize{
  \item{scores}{: matrix n rows is the number of individuals, k columns is the number of principal components}
\item{dbcluster}{: vector length n is the clustering results for each individual using dbscan}
\item{dbtable}{: table summary of the dbscan clustering results}
\item{kcluster}{: vector length n is the clustering results for each individual using kmeans}
\item{kmeantable}{: table summary of the kmeans clustering results}
\item{latentcurve}{: 3 D array n rows is the number of individuals, T columns is the total number of observational points, l: Total category Q minus 1 latent curves}
\item{meanfn}{: lists Q minus 1 vectors each has length T. The mean function for each one of the latent curves}
\item{eigenvalue}{: vector length k is each one of the eigen values from multivariate functional principal component analysis}
\item{eigenfn}{: matrix T rows is the number of total observational points, k columns is the number of principal components}
\item{probcurve}{: 3 D array n rows is the number of individual, T columns is the number of observational times, Q: the number of categories}
  }


}



\references{
Ramsay, J. O., Silverman, B. W. (2005). Functional Data Analysis. Springer. ISBN: 9780387400808

Happ, Clara, and Sonja Greven. Multivariate functional principal component analysis for data observed on different (dimensional) domains. Journal of the American Statistical Association 113.522 (2018): 649-659.
}
\author{

Xiaoxia Champon,  Ana-Maria Staicu, Anthony Weishampel, William Rand
}


\seealso{

Clustering of categorical valued functional data with application to social media
}
\examples{
catclust=catfdcluster(matrix(sample(c(0,1,2),100*250,replace=TRUE),nrow=100,ncol=250),seq(0,1,length=250),25,3,3,0.9,4,5,2,"happ","two")
}




