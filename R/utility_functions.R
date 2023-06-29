#######################################################################
# Purpose: Utility functions before clustering categorical functional data
# Author: Xiaoxia Champon
# Date: 1/21/2023
# Update:
#####################################################################

#required packages
library(fda)
library(refund)
library(mgcv)
library(funData)
library(MFPCA)
library(dbscan)
library(fossil)
library(NbClust)
library(ggplot2)
library(stats)
library(graphics)

#Function to return the logit
logit <- function(x){
  return(log(x/(1-x)))
}

#get smoothed curves
regression_g = function(z, Curves, tt, k=25, method="ML"){   #changed from 10 to 25
  z1 = Curves[z,]
  gam1 <- mgcv::gam(z1~s(tt, bs = "cr", m=2, k = k),
                    family="binomial", method = method,control=list(maxit = 500,mgcv.tol=1e-4,epsilon = 1e-04),optimizer=c("outer","bfgs"))
  return(gam1$fitted.values)
}

#original before 4/6/2022
##
#Step 1 of the proposed method in Anthony paper one: smoothing using link function
##

#New function to output predicted L-1 latent curves:
#input observed X_i1 or X_i2 binary curves and return smoothed Z_i1hat, Z_i2hat
Z_ihat=function(Curves_train,tt){
  N_train=dim(Curves_train)[1]
  vec = matrix(1:(N_train), ncol = 1)
  smoothed_x = logit(t(apply(vec, 1, function(x) regression_g(x, Curves_train, tt))))
  smoothed_x
}

phatc=function(Curves_train,tt){
  N_train=dim(Curves_train)[1]
  vec = matrix(1:(N_train), ncol = 1)
  smoothed_p = t(apply(vec, 1, function(x) regression_g(x, Curves_train, tt)))
  smoothed_p
}

mfundata=function(ufdata,t){
  mvdata=funData::funData(argvals = list(t), X = ufdata)
  mvdata
}

phatf=function (Zlatent) {
  numcat = dim(Zlatent)[3]+1
  n = dim(Zlatent)[1]
  nt = dim(Zlatent)[2]
  phatarray = array(0, dim = c(n, nt, numcat))
  denomarray = array(0, dim = c(n, nt, (numcat - 1)))
  for (i in 1:(numcat - 1)) {
    denomarray[, , i] = exp(Zlatent[, , i])

  }
  demsum = 1 + apply(denomarray, c(1, 2), sum)
  for (i in 1:(numcat - 1)) {
    phatarray[, , i] = exp(Zlatent[, , i])/demsum
  }
  sump = apply(phatarray, c(1, 2), sum)
  phatarray[, , numcat] = 1 - sump
  return(phat = phatarray)
}

##
score=function(mu,sd){
  rnorm(1,mean=mu,sd=sd)
}

#n number of subjects
#datapoints
#sparse=1 yes   sparse=0 no
#scorevar=2 bigger var , scorevar=1 smaller var
#ps=1 find z1,z2,z3, find p1,p2,p3, logp1-logp3
#ps=2 find z1,z2,z3, find p1,p2,p3=1-p1-p2 logp1-logp3
#ps=3 find z1,z2 staicu find z1hat z2hat
#ps=4 find z1,z2 staicu find z1hat z2hat but only use numerator
#k  #number of eigen functions
#q  #level of the categorical level

generate_data_scenario=function(k,n,datapoints,sparse,scorevar,ps,seed=123,st,et){
  k=k
  seed=seed
  st=st
  et=et
  scorevar=scorevar
  #k=3  #number of eigen functions
  q=3  #level of the categorical level

  if(sparse==1){
    mu_1=function(t){
      3.8+4*t  #-0.64+4*t    #4.5+4*t   #sparse 3.8+4*t

    }
    mu_2=function(t){
      1.5+4*t^2    #0.97+6*t^2

    }




    p_ihat=function(Z_i1app,Z_i2app){
      denom=(1+exp(Z_i1app)+exp(Z_i2app))
      p_i1h=exp(Z_i1app)/denom
      p_i2h=exp(Z_i2app)/denom
      p_i3h=1/denom
      # p_i3h=1-p_i1h- p_i2h
      return(list("p_1hatmatrix"=p_i1h,"p_2hatmatrix"=p_i2h,"p_3hatmatrix"=p_i3h))
    }
  }


  if (sparse==0){
    mu_1=function(t){
      #3.8+4*t
      -0.64+4*t    #4.5+4*t   #sparse 3.8+4*t

    }
    mu_2=function(t){
      #1.5+4*t^2
      0.97+6*t^2


    }

    p_ihat=function(Z_i1app,Z_i2app){
      denom=(1+exp(Z_i1app)+exp(Z_i2app))
      p_i1h=exp(Z_i1app)/denom
      p_i2h=exp(Z_i2app)/denom
      p_i3h=1- p_i1h- p_i2h
      return(list("p_1hatmatrix"=p_i1h,"p_2hatmatrix"=p_i2h,"p_3hatmatrix"=p_i3h))
    }

  }
  if (sparse==4){
    mu_1=function(t){
      #3.8+4*t
      #-0.64+4*t    #4.5+4*t   #sparse 3.8+4*t
      t+1

    }
    mu_2=function(t){
      #1.5+4*t^2
      #0.97+6*t^2
      #t-3
      t-1

    }

    p_ihat=function(Z_i1app,Z_i2app){
      denom=(1+exp(Z_i1app)+exp(Z_i2app))
      p_i1h=exp(Z_i1app)/denom
      p_i2h=exp(Z_i2app)/denom
      p_i3h=1- p_i1h- p_i2h
      return(list("p_1hatmatrix"=p_i1h,"p_2hatmatrix"=p_i2h,"p_3hatmatrix"=p_i3h))
    }

  }
  ###noise group
  if (sparse==5){
    mu_1=function(t){
      3.8+4*t
      #-0.64+4*t    #4.5+4*t   #sparse 3.8+4*t
      #rep(10,length(t))  #1

    }
    mu_2=function(t){
      #1.5+4*t^2
      0.97+6*t^2
      #t-3
      #rep(8,length(t))  #2

    }

    p_ihat=function(Z_i1app,Z_i2app){
      denom=(1+exp(Z_i1app)+exp(Z_i2app))
      p_i1h=exp(Z_i1app)/denom
      p_i2h=exp(Z_i2app)/denom
      p_i3h=1- p_i1h- p_i2h
      return(list("p_1hatmatrix"=p_i1h,"p_2hatmatrix"=p_i2h,"p_3hatmatrix"=p_i3h))
    }

  }


  mu_vec=rep(0,k)


  psi_fn=function(k){

    psi_k1=matrix(rep(1,length(t)*k),ncol=k)
    psi_k2=matrix(rep(1,length(t)*k),ncol=k)
    for (i in 1:k) {
      psi_k1[,i]=sin(2*i*pi*t )
      psi_k2[,i]=cos(2*i*pi*t )
    }
    list("psi_k1"=psi_k1,"psi_k2"=psi_k2)
  }


  t=seq(from = st,to = et, length=datapoints)

  X_i=array(0,dim=c(q,datapoints,n))  #multinormial results: row is level q, column is time points, n is the number of subjects, each column only has one row of 1 and every other rows are 0
  X_nt=matrix(rep(1,n*length(t)),nrow=n,ncol=length(t))  #true observations of categorical-valued outcome, each row represent one subject, columns represent time points
  score_matrix=matrix(rep(1,n*k),nrow=n,ncol=k)  #row is number of subjects, column is the number of eigen functions
  psi_score_matrix_1=matrix(rep(1,n*length(t)),ncol=n)  #dim: length(t)*nsubjects
  psi_score_matrix_2=matrix(rep(1,n*length(t)),ncol=n)
  Z_i1=matrix(rep(1,n*length(t)),nrow=n)  #True latent curves1:row is n subjects, col is t time points
  Z_i2=matrix(rep(1,n*length(t)),nrow=n) #True latent curve 2
  p_i1=matrix(rep(0,n*length(t)),nrow=n)  #True p_i1
  p_i2=matrix(rep(0,n*length(t)),nrow=n)  #True p_i2
  p_i3=matrix(rep(0,n*length(t)),nrow=n)  #True p_i3
  for (i in 1:n){
    set.seed(seed+i)

    if (k==3){
      if (scorevar==1){
        #score varies based on i
        score_1=score(0,1)
        score_2=score(0,sqrt(1/2))
        score_3=score(0,1/2)

      }

      if (scorevar==2){
        #score varies based on i
        score_1=score(0,1)
        score_2=score(0,sqrt(1/3))
        score_3=score(0,1/3)
      }

      if (scorevar==3){
        #score varies based on i
        score_1=score(0,1)
        score_2=score(0,sqrt(1/4))
        score_3=score(0,1/4)
      }


      if (scorevar==4){
        #score varies based on i
        score_1=score(-0.5,1)
        score_2=score(1,sqrt(1/2))
        score_3=score(0.25,1/2)
      }

      score_vector=cbind(score_1,score_2,score_3)
    }



    if (k==4){

      if (scorevar==1){
        score_1=score(0,1)
        score_2=score(0,sqrt(1/2))
        score_3=score(0,1/2)
        score_4=score(0,sqrt(1/8))
        # cpve=cumsum(c(1,sqrt(1/2),1/2,sqrt(1/8)))/sum(c(1,sqrt(1/2),1/2,sqrt(1/8)))
        # cvar=c(1,sqrt(1/2),1/2,sqrt(1/8))
      }



      if (scorevar==2){
        score_1=score(0,1)
        score_2=score(0,sqrt(1/3))
        score_3=score(0,1/3)
        score_4=score(0,sqrt(1/27))
        # cpve=cumsum(c(1,sqrt(1/3),1/3,sqrt(1/27)))/sum(c(1,sqrt(1/3),1/3,sqrt(1/27)))
        # cvar=c(1,sqrt(1/3),1/3,sqrt(1/27))
      }



      if (scorevar==3){
        score_1=score(0,1)
        score_2=score(0,sqrt(1/4))
        score_3=score(0,1/4)
        score_4=score(0,sqrt(1/64))
        # cpve=cumsum(c(1,sqrt(1/4),1/4,sqrt(1/64)))/sum(c(1,sqrt(1/4),1/4,sqrt(1/64)))
        # cvar=c(1,sqrt(1/4),1/4,sqrt(1/64))
      }
      score_vector=cbind(score_1,score_2,score_3,score_4)

    }







    psi_k1=psi_fn(k)$psi_k1
    psi_k2=psi_fn(k)$psi_k2

    #Z varies based on i
    #psi t*k, score: t*k,  psi%*%t(score)
    psi_score_matrix_1[,i]=psi_k1%*%t(score_vector)
    Z_i1[i,]=mu_1(t)+psi_score_matrix_1[,i]

    psi_score_matrix_2[,i]=psi_k2%*%t(score_vector)
    Z_i2[i,]=mu_2(t)+psi_score_matrix_2[,i]


    #p varies based on i
    denominator=(1+exp(as.vector(Z_i1[i,]))+exp(as.vector(Z_i2[i,])))
    p_i1[i,]=(exp(as.vector(Z_i1[i,])))/denominator
    p_i2[i,]=(exp(as.vector(Z_i2[i,])))/denominator
    p_i3[i,]=1-p_i1[i,]-p_i2[i,]


    #X_i varies based on i
    #X_i=matrix(rep(1,k*length(t)),nrow=k,ncol=length(t))

    for (j in 1:length(t)){
      X_i[,j,i]=rmultinom(n=1, size=1, prob=c(p_i1[i,j],p_i2[i,j],p_i3[i,j]))
    }

    #X_it varies based on i
    X_it=c(1)
    for (j in 1:length(t)){
      X_it[j]=as.vector(which(X_i[,j,i] == 1))
    }
    X_nt[i,]=X_it

    #collect score matrix
    score_matrix[i,]=score_vector
  }

  #collect value and graph
  #collect first two rows of observed binary curves
  X_i1=t(X_i[1,,])  #all n row subjects , t columns values related to p1
  X_i2=t(X_i[2,,]) #all n row subjects , t columns values related to p2
  X_i3=t(X_i[3,,]) #all n row subjects , t columns values related to p3

  #recover Z_i1 hat using X_i[1,all j, all n] only related to p1
  Z_i1hat=Z_ihat(X_i1,t)
  #recover Z_i2 hat using X_i[2,all j, all n] only related to p2
  Z_i2hat=Z_ihat(X_i2,t)
  Z_i3hat=Z_ihat(X_i3,t)



  if(ps==1){
    Z_i1hatstar=Z_i1hat+log(1+exp(Z_i3hat))-log(1+exp(Z_i1hat))-Z_i3hat
    Z_i2hatstar=Z_i2hat+log(1+exp(Z_i3hat))-log(1+exp(Z_i2hat))-Z_i3hat
  }


  if(ps==2){
    smooth_p1=(1+exp(-Z_i1hat))^(-1)
    smooth_p2=(1+exp(-Z_i2hat))^(-1)
    smooth_p3=1-smooth_p1-smooth_p2
    Z_i1hatstar=log(smooth_p1)-log(smooth_p3)
    Z_i2hatstar=log(smooth_p2)-log(smooth_p3)
  }

  if(ps==3){
    common_check=1-exp(Z_i1hat+Z_i2hat)-exp(Z_i2hat)
    common_check[common_check<=0]=0.001
    Z_i1hatstar=Z_i1hat+log(1+exp(Z_i2hat))-log(common_check)
    Z_i2hatstar=Z_i2hat+log(1+exp(Z_i1hat))-log(common_check)
  }



  if (ps==4){
    common_check=1-exp(Z_i1hat+Z_i2hat)-exp(Z_i2hat)
    common_check[common_check<=0]=0.001
    Z_i1hatstar=Z_i1hat-log(common_check)
    Z_i2hatstar=Z_i2hat+log(1+exp(Z_i1hat))-log(common_check)
  }



  p_i1hat=p_ihat(Z_i1hatstar,Z_i2hatstar)$p_1hatmatrix
  p_i2hat=p_ihat(Z_i1hatstar,Z_i2hatstar)$p_2hatmatrix
  p_i3hat=p_ihat(Z_i1hatstar,Z_i2hatstar)$p_3hatmatrix


  # truel=list("TrueX1"=X_i1,"TrueX2"=X_i2,"TrueX3"=X_i3,"TrueZ_i1"=Z_i1,"TrueZ_i2"=Z_i2,"Truep_i1"=p_i1,"Truep_i2"=p_i2,"Truep_i3"=p_i3,"Truescore_matrix"=score_matrix,"Truecatcurve"=X_nt)
  truel=list("TrueZ_i1"=Z_i1,"TrueZ_i2"=Z_i2,"Truep_i1"=p_i1,"Truep_i2"=p_i2,"Truep_i3"=p_i3,"Truescore_matrix"=score_matrix,"Truecatcurve"=X_nt,"comp1"=psi_k1,"comp2"=psi_k2)
  est=list("EstimateZ_i1"=Z_i1hatstar,"EstimateZ_i2"=Z_i2hatstar,"Estimatep_i1"=p_i1hat,"Estimatep_i2"=p_i2hat,"Estimatep_i3"=p_i3hat)
  return(list("Trueth"=truel,"Est"=est))
}


##function to visualize the clustering results
plot_cluster=function(scores_z,dbcluster,kcluster,st,et,datapoints){
  #DSCAN
  clusterdata=data.frame(scores_z)
  clusterdata$Cluster=as.factor(dbcluster)
  #clusterdata$Cluster=as.factor(res$cluster)
  colnames(clusterdata)[1:2] =c("ksi1","ksi2")

  tps <- ggplot2::ggplot(clusterdata,ggplot2::aes(ksi1,ksi2,colour = Cluster)) + ggplo2::geom_point(aes(shape=Cluster),size=3)+ggplot2::ggtitle(paste0("DBSCAN Cluster Results",'\n',"(",dim(clusterdata)[1]," Subjects",")")) +
    ggplot2::xlab(expression('Score '* widehat(xi[i1]))) + ggplot2::ylab(expression('Score '* widehat(xi[i2])))+ ggplot2::theme(plot.title = element_text(hjust = 0.5))+
    ggplot2::theme(text=element_text(size = 20))
  ###kmeans

  clusterdatak=data.frame(scores_z)
  clusterdatak$Cluster=as.factor(kcluster)
  colnames(clusterdatak)[1:2] =c("ksi1","ksi2")
  tpskmeans <- ggplot2::ggplot(clusterdatak,ggplot2::aes(ksi1,ksi2,colour = Cluster)) + ggplot2::geom_point(aes(shape=Cluster),size=3)+ggplot2::ggtitle(paste0("Kmeans Cluster Results",'\n',"(",dim(clusterdatak)[1]," Subjects",")")) +
    ggplot2::xlab(expression('Score '* widehat(xi[i1]))) + ggplot2::ylab(expression('Score '* widehat(xi[i2])))+ ggplot2::theme(plot.title = element_text(hjust = 0.5))+
    ggplot2::theme(text=element_text(size = 20))

  return(list("figdbscan"=tps,"figkmeans"=tpskmeans))
}

##generate simulate data
catsim=function(seed=123,datapoints,n,sparse1,sparse2,scorevar1,scorevar2,ps,st,et,k,propc1,propc2,propnoise){
  clustern50t250=generate_data_scenario(k=k,n=n*propc1,datapoints,sparse1,scorevar1,ps,seed=seed,st,et)
  clustern50t250p2=generate_data_scenario(k=k,n=n*propc2,datapoints,sparse2,scorevar2,ps,seed=seed,st,et)
  clustern50t250p3=generate_data_scenario(k=k,n=n*propnoise,datapoints,5,scorevar1,ps,seed=seed,st,et)

  truen100t250=mapply(rbind,clustern50t250$Trueth,clustern50t250p2$Trueth,clustern50t250p3$Trueth,SIMPLIFY=FALSE,USE.NAMES = TRUE)
  estn100t250=mapply(rbind,clustern50t250$Est,clustern50t250p2$Est,clustern50t250p3$Est,SIMPLIFY=FALSE,USE.NAMES = TRUE)

  combn100t250=list("Trueth"=truen100t250, "Est"=estn100t250)
  datainput=combn100t250$Trueth$Truecatcurve
  return("catdata"=datainput)
}

##plot distance
plot_kdistance=function(scores_z,knnum,pct){
  dist=dbscan::kNNdist(scores_z, k = knnum)
  distdata=data.frame(sort(dist))
  distdata$index=1:dim(distdata)[1]
  ninty5p=quantile(dist, probs = pct)
  dp <- ggplot2::ggplot(distdata,ggplot2::aes(index,sort.dist.)) + ggplot2::geom_line()+ggplot2::ggtitle(paste0(knnum,"-NN Distance Plot ",'\n',"(",dim(distdata)[1]," Subjects",")")) +
    ggplot2::xlab("Points sorted by Distance") + ggplot2::ylab("Distance")+ ggplot2::theme(plot.title = element_text(hjust = 0.5))+ggplot2::geom_hline(yintercept=ninty5p, color = "red")+
    ggpplot2::geom_text(data=data.frame(round(ninty5p,2)),
              ggplot2::aes(x=dim(distdata)[1]/2,y=1.2*ninty5p,label=paste0("Distance at ",gsub("%$","",row.names(data.frame(round(ninty5p,2)))),"th percentile= ",round(ninty5p,2))))
  return(list("kdistfig"=dp))
}


#plot latent curves and probability curves
#vectort=c(which(cluster==0))
#plot_latent=function(zlatent,phat,st,et,cluster,labelnum,argval,zmean){
plot_latent=function(zlatent,phat,st,et,cluster,labelnum,argval){
  vectort=c(which(cluster==labelnum))
  datapoints=dim(zlatent)[2]

  n=length(vectort)
  ########################################################################################

  #plot 1: truth and smoothed
  ########################################################################################
  #entire block is  n*t dimension
  #truth Z, P and smoothed Z, P
  numz=dim(zlatent)[3]
  nump=dim(phat)[3]
  zest=zlatent[vectort,,]
  pest=phat[vectort,,]

  #Z_i
  #meanz=colMeans(zest)   #t*numz
  # meanz=zmean
  #meanfnZ[[1]]@X
  for (i in 1:numz){
    matplot(argval, t(zest[,,i]),
            type='l', lty=1, col="light grey",
            #main=mtext(bquote("Estimated Latent Cruve "*widehat('Z')[i])),
            #xlab="Number of Datapoints", ylab=expression(widehat('Z')[i]))

            main=mtext(bquote("Estimated Latent Cruve ")),
            xlab="Time", ylab="Value")
    #lines(argval,meanz[,i] ,
    #type='l', lty=1, lwd=2, col = "red")
    #lines(argval,zmean[[i]]@X ,
    #type='l', lty=1, lwd=2, col = "red")


    # legend(x = "topleft",  #horiz = TRUE,        # Position
    #        legend = c("Individual", "Mean"),  # Legend texts
    #        #lty = c(1, 2),           # Line types
    #        col = c("grey", "red"),           # Line colors
    #        lwd = 2)

  }



  ###################################################################################
  #p_i

  meanp=colMeans(pest)   #t*nump

  for (i in 1:nump){
    matplot(argval, t(pest[,,i]),
            type='l', lty=1, col="light grey",
            #main=mtext(bquote("Estimated Latent Cruve "*widehat('p')[i])),
            #xlab="Number of Datapoints", ylab=expression(widehat('p')[i]))
            main=mtext(bquote("Estimated Probability Cruve ")),
            xlab="Time", ylab='Value')
    lines(argval,meanp[,i] ,
          type='l', lty=1, lwd=2, col = "red")
    # legend(x = "topright",          # Position
    #        legend = c("Individual", "Mean"),  # Legend texts
    #        #lty = c(1, 2),           # Line types
    #        col = c("grey", "red"),           # Line colors
    #        lwd = 2)                 # Line width
  }

}

#rand index from simulated data
randfn=function(dbcluster,kcluster,nsub){
  tlabel=c(rep(1,nsub*0.7),rep(2,nsub*0.3),rep(3,nsub*0.04))

  randdbscan=rand.index(tlabel,dbcluster)
  randkmeans=rand.index(tlabel,kcluster)
  return(list("randd"=randdbscan,"randk"=randkmeans))
}
####elbow method
elbow=function (data, plot = TRUE) {
    if (missing(data)) {
        stop("Please provide a two-columns data frame.")
    }
    if (!is.list(data)) {
        stop("`data` must be a two-columns data frame.")
    }
    if (ncol(data) > 2) {
        warning("Only the first two columns will be considered.")
    }
    if (!is.numeric(data[, 1]) || !is.numeric(data[, 2])) {
        stop("Non-numeric data detected.")
    }
    if (sum(is.na(data[, 1])) + sum(is.na(data[, 2]))) {
        stop("Missing values detected.")
    }
    if (!is.logical(plot)) {
        stop("`plot` must be a boolean.")
    }
    data <- data[, 1:2]
    data <- data[order(data[, 1]), ]
    constant <- data[c(1, nrow(data)), ]
    colnames(constant) <- c("x", "y")
    mod <- stats::lm(y ~ x, data = constant)
    data[, "constant"] <- round(mod$coef[[1]] + mod$coef[[2]] * 
        data[, 1], 3)
    pos <- round(nrow(data)/2)
    if (data[pos, "constant"] < data[pos, 2]) {
        ymin <- min(data[, 2])
        data[, "benefits"] <- ymin + round(data[, 2] - data[, 
            "constant"], 3)
        maxi <- data[which.max(data[, "benefits"]), ]
    }
    else {
        ymax <- max(data[, 2])
        data[, "benefits"] <- ymax - round(data[, "constant"] - 
            data[, 2], 3)
        maxi <- data[which.min(data[, "benefits"]), ]
    }
    xxx <- list()
    xxx[[1]] <- maxi[1, 1]
    xxx[[2]] <- data
    names(xxx) <- c(paste(colnames(data)[1], "selected", sep = "_"), 
        "data")
    if (plot) {
        xlims <- range(data[, 1])
        ylims <- c(min(c(data[, 2], data[, 3], data[, 4])), max(data[, 
            2]))
        graphics::par(mar = c(2.5, 2.5, 1.5, 1.5), family = "serif", 
            cex.axis = 0.85, mgp = c(2, 0.15, 0), tcl = -0.25, 
            fg = "#666666", col = "#666666", col.axis = "#666666")
        graphics::plot(x = data[, 1], y = data[, 2], xlim = xlims, 
            ylim = ylims, ann = FALSE, axes = FALSE, type = "n")
        graphics::grid()
        graphics::box()
        graphics::par(mgp = c(2, 0, 0))
        graphics::axis(1, lwd = 0)
        graphics::axis(1, maxi[, 1], lwd = 0, font = 2, col.axis = "black")
        graphics::par(mgp = c(2, 0.25, 0))
        graphics::axis(2, lwd = 0, las = 1)
        at <- round(maxi[, 2], 3)
        graphics::axis(2, at, lwd = 0, font = 2, col.axis = "black", 
            las = 1)
        graphics::mtext(side = 1, cex = 1, line = 1.25, text = expression("x"))
        graphics::mtext(side = 2, cex = 1, line = 1.45, text = expression("f(x)"))
        graphics::polygon(x = c(data[, 1], data[1, 1]), y = c(data[, 
            "benefits"], data[1, "benefits"]), col = "#aaaaaa66", 
            border = "#aaaaaa")
        graphics::points(x = data[, 1], y = data[, 2], type = "b", 
            pch = 19, col = "black")
        graphics::lines(x = rep(maxi[1, 1], 2), y = c(par()$usr[3], 
            maxi[1, 2]), col = "black", lwd = 0.5)
        graphics::lines(x = c(par()$usr[1], maxi[1, 1]), y = rep(maxi[1, 
            2], 2), col = "black", lwd = 0.5)
        graphics::points(x = maxi[, 1], y = maxi[, 2], type = "b", 
            pch = 19, cex = 1.5, col = "black")
        graphics::points(x = maxi[, 1], y = maxi[, 4], type = "b", 
            pch = 19, cex = 1, col = "#666666")
    }
    return(xxx)
}


