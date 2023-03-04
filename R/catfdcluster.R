#######################################################################
# Purpose: clustering categorical functional data
# Author: Xiaoxia Champon
# Date: 1/21/2023
# Update:
#####################################################################


#source("utility_functions.R")
##input categorical functional data n*t and output clustering results, latent curves, probability curves
#catfd=matrix(sample(c(0,1,2),250*100,replace = TRUE),nrow=100)


##input categorical functional data n*t and output clustering results, latent curves, probability curves

#catfd=matrix(sample(c(0,1,2),250*100,replace = TRUE),nrow=100)

catfdcluster=function(catfd,argval,splines1D,M,knnum,pct,minPts,max.nc,min.nc,method,numdim){
  st=min(argval)
  et=max(argval)
  datapoints=dim(catfd)[2]
  tolcat=table(catfd)
  catorder=order(tolcat,decreasing = TRUE)
  numcat=length(catorder)
  refcat=catorder[numcat]
  refmat=catfd
  refmat[refmat!=refcat]=0
  refmat[refmat==refcat]=1
  nsub=dim(catfd)[1]
  ntime=dim(catfd)[2]
  subdata=array(data=NA,c(nsub,ntime,numcat))
  for (i in 1:numcat){
    datacopy=catfd
    datacopy[datacopy!=i]=0
    datacopy[datacopy==i]=1
    subdata[,,i]=datacopy
  }
  t=seq(st,et,length=datapoints)
  #input observed X_i1 or X_i2 binary curves and return smoothed Z_i1hat, Z_i2hat
  Zihat=array(data=NA,c(nsub,ntime,numcat))
  for (i in 1:numcat){
    datacopy=subdata
    Zihat[,,i]=Z_ihat(datacopy[,,i],t)
  }

  Zihatstar=array(data=NA,c(nsub,ntime,numcat-1))
  for (i in 1:(numcat-1)){
    datacopy=Zihat
    Zihatstar[,,i]=Zihat[,,i]+log(1+exp(Zihat[,,numcat]))-log(1+exp(Zihat[,,i]))-Zihat[,,numcat]
  }

  phatmat=phatf(Zihatstar)



  #############################################
  if (method=="ramsay"){
    fdarange = c(st, et)
    fdabasis = fda::create.bspline.basis(fdarange, splines1D, 4)
    fdatime = seq(st, et, length = datapoints)
    Zihatstarnew = apply(Zihatstar, c(1, 3), t)
    fdafd = fda::smooth.basis(fdatime, Zihatstarnew, fdabasis)$fd
    nharm = M
    fdapcaList = fda::pca.fd(fdafd, nharm)

    finalM=which(cumsum(fdapcaList$values)/sum(fdapcaList$values)>=0.95)[1]
    fdapcaListfinal = fda::pca.fd(fdafd, finalM)
    scores_z = apply(fdapcaListfinal$scores, c(1, 2), sum)
    values=fdapcaListfinal$values
    meanfn=fdapcaList$meanfd
    eigenfn=fdapcaList$harmonics

    dims=dim(scores_z)[2]
    if (dims<2){
      #finalM=which(cumsum(fdapcaList$values)/sum(fdapcaList$values)>=0.98)[1]
      finalM=2
      fdapcaListfinal = fda::pca.fd(fdafd, finalM)
      scores_z = apply(fdapcaListfinal$scores, c(1, 2), sum)
      values=fdapcaListfinal$values
      meanfn=fdapcaList$meanfd
      eigenfn=fdapcaList$harmonics
    }

  }

  if (method=="happ"){
    vecapply=matrix(1:(dim(Zihatstar)[3]),ncol=1)
    mfdataused=apply(vecapply,1,function(x) {mfundata(Zihatstar[,,x],t)})
    mvdata=funData::multiFunData(mfdataused)


    uniexpan=list()
    # MFPCA based on univariate FPCA Z_ihat
    for (i in 1:(numcat-1)){
      uniexpan[[i]]=list(type = "splines1D", k = splines1D)
    }

    # MFPCA based on univariate FPCA Z_ihat
    uFPCA <- MFPCA::MFPCA(mvdata, M = M, uniExpansions = uniexpan)

    finalM=which(cumsum(uFPCA$values)/sum(uFPCA$values)>=0.95)[1]
    uFPCA <- MFPCA::MFPCA(mvdata, M = finalM, uniExpansions = uniexpan)
    scores_z=uFPCA$scores
    values=uFPCA$values
    eigenfn=uFPCA$functions
    meanfn=uFPCA$meanFunction
  }

  #7, 0.98 2 clusters  6, 88
  #res <- dbscan::dbscan(scores_z, eps =pct , minPts = minPts )
  dimz=dim(scores_z)[2]
  if (dimz<=2){minPts=4}
  if (dimz>2){minPts=2*dimz+1}
  #dist=kNNdist(combinedscore_z, k = knnum)
  dist=dbscan::kNNdist(scores_z, k = minPts)
  ninty5p=quantile(dist, probs = pct)

  #########change to max increase
  #sortdist=sort(dist)
  #epsoptimal=sortdist[which.max(diff(sortdist))]
  ##############################

  #########################################################
  #dist=dbscan::kNNdist(scores_z, k = knnum)
  distdata=data.frame(sort(dist))
  distdata$index=1:dim(distdata)[1]
  ninty5p=quantile(dist, probs = pct)
  #dp <- ggplot2::ggplot(distdata,ggplot2::aes(index,sort.dist.)) + geom_line()+ggtitle(paste0(knnum,"-NN Distance Plot ",'\n',"(",dim(distdata)[1]," Subjects",")")) +
  #xlab("Points sorted by Distance") + ylab("Distance")+ theme(plot.title = element_text(hjust = 0.5))+geom_hline(yintercept=ninty5p, color = "red")+

  #geom_text(data=data.frame(round(ninty5p,2)), ggplot2::aes(x=dim(distdata)[1]/2,y=1.2*ninty5p,label=paste0("Distance at ",gsub("%$","",row.names(data.frame(round(ninty5p,2)))),"th percentile= ",round(ninty5p,2))))
  ##############################################

  if (numdim=="all"){
    res <- dbscan::dbscan(scores_z, eps =ninty5p , minPts = minPts)
    #res <- dbscan::dbscan(scores_z, eps =epsoptimal , minPts = minPts)

    clustertable=table(res$cluster)
    #tclustertable    #z score

    #########Kmeans
    reskmeans=NbClust::NbClust(data = scores_z, diss = NULL, distance = "euclidean",
                               min.nc = min.nc, max.nc = max.nc, method = "kmeans")
    clustertablek=table(reskmeans$Best.partition)
  }
  if (numdim=="two"){

    res <- dbscan::dbscan(scores_z[,1:2], eps =ninty5p , minPts = minPts)

    clustertable=table(res$cluster)
    #tclustertable    #z score

    #########Kmeans
    reskmeans=NbClust::NbClust(data = scores_z[,1:2], diss = NULL, distance = "euclidean",
                               min.nc = min.nc, max.nc = max.nc, method = "kmeans")
    clustertablek=table(reskmeans$Best.partition)
  }


  #return(list("scores"=scores_z,"dbcluster"=res$cluster,"dbtable"=clustertable,
  #"kcluster"=reskmeans$Best.partition,"kmeantable"=clustertablek,
  #"latentcurve"=Zihatstar,"meanfn"=uFPCA$meanFunction,"eigenvalue"=uFPCA$values,
  #"eigenfn"=uFPCA$functions,"probcurve"=phatmat))


  return(list("scores"=scores_z,"dbcluster"=res$cluster,"dbtable"=clustertable,
              "kcluster"=reskmeans$Best.partition,"kmeantable"=clustertablek,
              "latentcurve"=Zihatstar,"eigenvalue"=values,"meanfn"=meanfn,
              "eigenfn"=eigenfn,"probcurve"=phatmat))
}


