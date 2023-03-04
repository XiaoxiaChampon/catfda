#######################################################################
# Purpose: simulate data and clustering categorical functional data sample code
# Author: Xiaoxia Champon
# Date: 1/21/2023
# Update:
#####################################################################

#library(fda)
#library(refund)
#library(mgcv)
#library(funData)
#library(MFPCA)
#library(dbscan)
#library(fossil)
#library(NbClust)
#library(ggplot2)


#######################################################################################################
#test the code (need to uncomment and run the example code)
#catsim=function(seed=123,datapoints,n,sparse1,sparse2,scorevar1,scorevar2,ps,st,et,k,propc1,propc2,propnoise)
######################################################################################################
#block 1
#dpoints=1000
#nsub=100
#knnum=3
#pct=0.9
#testcatdata=catsim(seed=123,dpoints,nsub,4,1,1,1,1,0.01,pct,3,0.7,0.3,0.04)


#catcluster=function(catfd,st,et,splines1D,M,knnum,pct,minPts,max.nc,min.nc,numdim)
######################################################################################
#block 2
#st=Sys.time()
#testcluster=catfdcluster(testcatdata,0,1,25,3,3,0.99,3,5,2,"two")
#et=Sys.time()
#telapse=et-st
#telapse
#not needed anymore ####testcluster$distfig
#####################################################################################

#randfn=function(dbcluster,kcluster,nsub)
####################################################################################
#block 3
#randfn(testcluster$dbcluster,testcluster$kcluster,nsub)
######################################################################################
#plot_cluster=function(scores_z,dbcluster,kcluster,st,et,datapoints)
#block5
#plot_cluster(testcluster$scores,testcluster$dbcluster,testcluster$kcluster,0,1,dpoints,knnum,pct)
########################################################################################
#plot_latent=function(zlatent,phat,st,et,cluster,labelnum,argval,knnum,pct)
######################################################################################
#block 6
#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$dbcluster,0,seq(0,1,length=dpoints))  #testcluster$meanfn

#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$dbcluster,1,seq(0,1,length=dpoints))
#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$dbcluster,2,seq(0,1,length=dpoints))

#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$kcluster,3,seq(0,1,length=dpoints))

#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$kcluster,1,seq(0,1,length=dpoints))

#plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#testcluster$kcluster,2,seq(0,1,length=dpoints))

# #catcluster=function(catfd,st,et,splines1D,M,knnum,pct,minPts,max.nc,min.nc)
# st=Sys.time()
# testcluster=catfdcluster(testcatdata,0,1,25,3,3,0.99,3,5,2)
# et=Sys.time()
# telapse=et-st
# telapse
# testcluster$distfig

#randfn=function(dbcluster,kcluster,nsub)
#randfn(testcluster$dbcluster,testcluster$kcluster,nsub)

#plot_cluster=function(scores_z,dbcluster,kcluster,st,et,datapoints)
#plot_cluster(testcluster$scores,testcluster$dbcluster,testcluster$kcluster,0,1,dpoints)

#plot_latent=function(zlatent,phat,st,et,cluster,labelnum,argval)
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$dbcluster,0,seq(0,1,length=dpoints))  #testcluster$meanfn
#
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$dbcluster,1,seq(0,1,length=dpoints))
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$dbcluster,2,seq(0,1,length=dpoints))
#
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$kcluster,3,seq(0,1,length=dpoints))
#
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$kcluster,1,seq(0,1,length=dpoints))
#
# plot_latent(testcluster$latentcurve,testcluster$probcurve,0,1,
#             testcluster$kcluster,2,seq(0,1,length=dpoints))

#####################################################################################################
