library(profvis)

library(mgcv)

library(foreach)
library(doParallel)


# n.cores <- parallel::detectCores() - 1
# my.cluster <- parallel::makeCluster(n.cores, type = "PSOCK")
# doParallel::registerDoParallel(cl = my.cluster)
# cat("Parellel Registered: ", foreach::getDoParRegistered())
#
# set.seed(123)
# x <- foreach(indv = 1:10, .combine=rbind) %do% {
#   # list("n"=array(rnorm(6), c(2,3)), "a"=sqrt(indv))
#   # c(indv,sqrt(indv),indv*indv)
#   cbind(c(1:10) * indv * 1.0, array(30 + indv,10))
# }
# dim(x)
# typeof(x)
# x
# array(x)
#
#
# set.seed(123)
# zz <- NULL
# pp <- array(0, c(10, 20))
# for(indv in 1:10)
# {
#   pp[indv,] <- cbind(c(1:10) * indv, c(31:40))
#   # zz <- cbind(zz, c(1:10) * indv)
# }
# typeof(pp)
# pp
# array(pp)

# print(x)
#
# x1 <- as.vector(x[,1])
# x2 <- as.vector(x[,2])
# x3 <- as.vector(x[,3])
# print(x1)
# print(x2)
# print(x3)
#
# parallel::stopCluster(cl = my.cluster)

# profvis({

ClusterSimulation <- function(num_indvs, timeseries_length,
                              scenario, num_replicas,
                              seed_cluster=1230, seed_cfd=9876)
{
  cat("Cluster Simulation\nNum Indvs:\t", num_indvs,
      "\nTimeseries Len:\t", timeseries_length,
      "\nNum Replicas:\t", num_replicas)

  occur_fraction <- GetOccurrenceFractions(scenario)
  cat("\nOccurrence Fractions: ", occur_fraction)

  cluster_allocation <- occur_fraction * num_indvs
  cat("\nIndividuals in each cluster (cluster alloc.): ", cluster_allocation)

  true_cluster <- rep(c(1,2,3), cluster_allocation)
  cat("\nTrue Cluster:\n", true_cluster)

  true_cluster_db <- rep(c(1,2,0), cluster_allocation)
  cat("\nTrue Cluster DB:\n", true_cluster_db)

  for(replica_idx in 1:num_replicas)
  {
    cat("\nReplica: ", replica_idx)

    # generate clusters
    set.seed(seed_cluster + 100 * replica_idx)
    cat("\nCluster", replica_idx, " --> seed: ", seed_cluster + 100 * replica_idx)

    cluster_f1 <- GenerateClusterData(1, scenario, 3, cluster_allocation[1], timeseries_length)
    cluster_f2 <- GenerateClusterData(2, scenario, 3, cluster_allocation[2], timeseries_length)
    cluster_f3 <- GenerateClusterData(3, scenario, 3, cluster_allocation[3], timeseries_length)

    # Recover the latent Gaussian process --> is this always 2 ???
    Z1 <- cbind(cluster_f1$Z1, cluster_f2$Z1, cluster_f3$Z1)
    Z2 <- cbind(cluster_f1$Z2, cluster_f2$Z2, cluster_f3$Z2)

    # Recover the true probability curves --> could there be more than 3 ???
    prob_curves <- list(p1=cbind(cluster_f1$p1, cluster_f2$p1, cluster_f3$p1),
                        p2=cbind(cluster_f1$p2, cluster_f2$p2, cluster_f3$p2),
                        p3=cbind(cluster_f1$p3, cluster_f2$p3, cluster_f3$p3))

    # generate categFuncData
    set.seed(seed_cfd + 100 * replica_idx)
    cat("\nCategFD", replica_idx, " --> seed: ", seed_cfd + 100 * replica_idx)

    categ_func_data_list <- GenerateCategFuncData(prob_curves)

    # what is Q vals ? better name???
    Q_vals <- unique(c(categ_func_data_list$W))
    if(is.numeric(Q_vals))
    {
      Q_vals <- sort(Q_vals)
    }

    # I need to know what this loop is meant to do !??? maybe there is a better way
    for(indv in 1:num_indvs)
    {
      if(indv %in% 1:cluster_allocation[1])
      {
        setting_choice <- 1
      }
      if(indv %in% (cluster_allocation[1] + 1):(cluster_allocation[1] + cluster_allocation[2]))
      {
        setting_choice <- 2
      }
      if(indv %in% (cluster_allocation[1] + cluster_allocation[2] + 1):num_indvs)
      {
        setting_choice <- 3
      }

      # better names for the following variables ???
      # 1. check weather one category only appears 1 time and is it in the end of the timeseries
      # 2. OR is it appearing only one time in the begining
      tolcat <- table(categ_func_data_list$W[, indv])
      catorder <- order(tolcat, decreasing = TRUE)
      numcat <- length(catorder)
      refcat <- catorder[numcat]
      count_iter <- 0
      while ((min(as.numeric(tolcat)) == 1 && categ_func_data_list$W[, indv][timeseries_length] == refcat && count_iter < 100)
             || (min(as.numeric(tolcat)) == 1 && categ_func_data_list$W[, indv][1] == refcat && count_iter < 100))
      {
        count_iter <- count_iter + 1
        new_cluster_data <- GenerateClusterData(setting_choice, "A", 3, 5, timeseries_length)

        new_prob_curves <- list(p1 = new_cluster_data$p1, p2 = new_cluster_data$p2, p3 = new_cluster_data$p3)
        new_categ_func_data_list <- GenerateCategFuncData( new_prob_curves )

        # what is this 3 ?? arbitrarily chosen?
        categ_func_data_list$W[, indv] <- new_categ_func_data_list$W[, 3]
        Z1[, indv] <- new_cluster_data$Z1[, 3] # latent curves Z1 and Z2
        Z2[, indv] <- new_cluster_data$Z2[, 3]

        categ_func_data_list$X[indv, , ] <- 0
        for (this_time in 1:timeseries_length)
        {
          categ_func_data_list$X[indv, this_time, which(Q_vals == categ_func_data_list$W[, indv][this_time])] <- 1
        }

        tolcat <- table(categ_func_data_list$W[, indv])
        catorder <- order(tolcat, decreasing = TRUE)
        numcat <- length(catorder)
        refcat <- catorder[numcat]
      } # end while
    } # end for(indv in 1:num_indvs)

    #Estimation
    timestamps01 <- seq(from = 0.0001, to = 1, length=timeseries_length)
    categFD_est <- EstimateCategFuncData(timestamps01, X=categ_func_data_list$X)
return(categFD_est)
    break # <-----------------REMOVE THIS //////////////////////////////////////////////////////////////

  } # END of "for(replica_idx in 1:num_replicas)'

  # # recover latent process
  # Z1_est=categFD_est$Z1_est
  # Z2_est=categFD_est$Z2_est
  #
  # #recover =probabilities for each category
  # p1_est<-categFD_est$p1_est
  # p2_est<-categFD_est$p2_est
  # p3_est<-categFD_est$p3_est
  #
  # if ( n==100 && scenario=="A")
  # {
  #   # evaluate performance Z and P
  #   rmse1_temp <- c(by(mse_bw_matrix(Z1,Z1_est) , true_cluster, mean))
  #   rmse2_temp <- c(by(mse_bw_matrix(Z2,Z2_est), true_cluster, mean))
  #   rmse[ii, ,] <- rbind(rmse1_temp,rmse2_temp )
  #
  #   error.p1<- mse_bw_matrixp(p1,p1_est)
  #   error.p2<- mse_bw_matrixp(p2,p2_est)
  #   error.p3<- mse_bw_matrixp(p3,p3_est)
  #
  #
  #   hellinger[ii, ,] <-  rbind( c(by(error.p1, true_cluster, mean)),
  #                               c(by(error.p2, true_cluster, mean)),
  #                               c(by(error.p3, true_cluster, mean)))
  #
  # }

  cat("\n")
  return (categFD_est)
}

#' Runs gam on the given data and returns results
#' Fits a binomial to describe the given in_x
#' @param timestamps01 current time value
#' @param in_x binary series
#' @return fit values and linear predictors both with length of time_series length
RunGam <- function(timestamps01, in_x, basis_size, method)
{
  basis_size_rev_1 <- max(min(round(min(sum(in_x), sum(1-in_x))/2), basis_size ), 5)

  fit_binom_1 <- gam(in_x~s(timestamps01, bs = "cr", m=2, k = basis_size_rev_1),
                     family=binomial(link="probit"), method = method,
                     control=list(maxit = 500,mgcv.tol=1e-4,epsilon = 1e-04),
                     optimizer=c("outer","bfgs"))
  p1 <- fit_binom_1$fitted.values
  p1_linpred <- fit_binom_1$linear.predictors

  return(list(prob=p1, linpred=p1_linpred))
}

EstimateCategFuncData <- function(timestamps01, X=NULL, W=NULL, basis_size=25, method="ML", sim=TRUE)
{
  if(is.null(X)) sim<-FALSE

  if(!sim)
  {
    num_indv<- ncol(W)
    timeseries_length <-nrow(W)
    category_count<- length(unique(c(W)))
    Q_vals <- unique(c(W))
    if(is.numeric(Q_vals)) Q_vals<- sort(Q_vals)

    X<- array(0, c(num_indv,timeseries_length,category_count))
    for(indv in 1:num_indv)
    {
      for(timestamps01 in 1:timeseries_length)
      {
        X[indv, timestamps01, which(Q_vals==W[, indv][timestamps01])] <- 1
      }
    }
  }

  num_indv<- dim(X)[1]
  timeseries_length<- dim(X)[2]
  category_count <- dim(X)[3]

  Z<-NULL
  p<-array(0, c(num_indv, timeseries_length, category_count))
  ##########################
  ###########################
  # num_indv is the subject and this step is done by subject level
  # can parallel
  #############################
  #############################
#   for (indv in 1:num_indv)
#   {
#     x1<- X[indv,,1]
#     x2<- X[indv,,2]
#     x3<- X[indv,,3]
#
#     # fit the Binom model
#     ###################################################
#     ###################################################
#     ##updated estimation
#
#     gam_result_1 <- RunGam(timestamps01, x1, basis_size, method)
#     p1 <- gam_result_1$prob
#     p1_linpred <- gam_result_1$linpred
#
#     gam_result_2 <- RunGam(timestamps01, x2, basis_size, method)
#     p2 <- gam_result_2$prob
#     p2_linpred <- gam_result_2$linpred
#
#     gam_result_3 <- RunGam(timestamps01, x3, basis_size, method)
#     p3 <- gam_result_3$prob
#     p3_linpred <- gam_result_3$linpred
#
#     # estimate the latent tranjecotries Z
#     exp_p3_linepred <- exp(p3_linpred)
#     z1 <- (p1_linpred-p3_linpred)-log( (1+exp(p1_linpred))/(1+exp_p3_linepred))
#     z2 <- (p2_linpred-p3_linpred)-log( (1+exp(p2_linpred))/(1+exp_p3_linepred))
#
#     Z <- cbind(Z, c(z1,z2))
#     psum <- (p1+p2+p3)
#     p[indv,,] <- cbind(p1/psum, p2/psum, p3/psum)
#   }
# return(p)
# return(list(Z=Z,p=p))

  zp <- foreach (indv = 1:num_indv, .combine = cbind, .init = NULL) %do%
  {
    x1<- X[indv,,1]
    x2<- X[indv,,2]
    x3<- X[indv,,3]

    # fit the Binom model
    ###################################################
    ###################################################
    ##updated estimation

    gam_result_1 <- RunGam(timestamps01, x1, basis_size, method)
    p1 <- gam_result_1$prob
    p1_linpred <- gam_result_1$linpred

    gam_result_2 <- RunGam(timestamps01, x2, basis_size, method)
    p2 <- gam_result_2$prob
    p2_linpred <- gam_result_2$linpred

    gam_result_3 <- RunGam(timestamps01, x3, basis_size, method)
    p3 <- gam_result_3$prob
    p3_linpred <- gam_result_3$linpred

    # estimate the latent tranjecotries Z
    exp_p3_linepred <- exp(p3_linpred)
    z1 <- (p1_linpred-p3_linpred)-log( (1+exp(p1_linpred))/(1+exp_p3_linepred))
    z2 <- (p2_linpred-p3_linpred)-log( (1+exp(p2_linpred))/(1+exp_p3_linepred))
    psum <- p1 + p2 + p3

    # Z <- cbind(Z, c(z1,z2))

    # p[indv,,] <- cbind(p1/psum, p2/psum, p3/psum)
    # return(cbind(p1/psum, p2/psum, p3/psum))
    return(c(c(z1,z2), cbind(p1/psum, p2/psum, p3/psum)))
  }
  # Unravel the two variables from zp
  z_rows_count <- timeseries_length * 2
  Z <- array(zp[1:z_rows_count, ], c(z_rows_count, num_indv))
  p <- array(t(matrix(zp[(z_rows_count + 1):dim(zp)[1], ], ncol=num_indv)), c(num_indv, timeseries_length, category_count))

  return(list(Z1_est=Z[1:timeseries_length,],
              Z2_est=Z[1:timeseries_length+timeseries_length,],
              p1_est=t(p[,,1]),
              p2_est=t(p[,,2]),
              p3_est=t(p[,,3]) ))
}

GenerateCategFuncData <- function(prob_curves)
{
  curve_count <- length(prob_curves);

  # we could have just passed these arguments ???
  num_indvs <- ncol(prob_curves$p1)
  timeseries_length <- nrow(prob_curves$p1)

  # better names for W and X ???
  W <- matrix(0, ncol=num_indvs, nrow=timeseries_length)
  X_array <- array(0, c(num_indvs, timeseries_length, curve_count))

  for(indv in c(1:num_indvs))
  {
    X <- sapply(c(1:timeseries_length),
                function(this_time) rmultinom(n=1,
                                      size=1,
                                      prob = c(prob_curves$p1[this_time,indv],
                                               prob_curves$p2[this_time,indv],
                                               prob_curves$p3[this_time,indv]) ))
    W[,indv] <- apply(X, 2, which.max)
    X_array[indv,,] <- t(X)
  }

  return(list(X=X_array, W=W)) # X_binary W_catfd
}

#' Get clustered data
#'
#'
GenerateClusterData <- function(setting, scenario, k, num_indvs, timeseries_length)
{
  setting_object <- GetMuAndScore(setting, scenario, k)
  cluster_f <- GenerateClusterDataScenario(num_indvs,
                                 timeseries_length,
                                 k,
                                 mu_1 = setting_object$mu_1,
                                 mu_2 = setting_object$mu_2,
                                 score_vals = setting_object$score_vals)
  return (cluster_f)
}

#' Get fraction of occurrence of each class for a given scenario
#' @param scenario scenario name as a string "A", "B", "C"
#' @return a vector containing the fractions
#'
GetOccurrenceFractions <- function(scenario)
{
  occur_fraction <- switch (scenario,
                            "A" = c(0.75, 0.22, 0.03),
                            "B" = c(0.5, 0.3, 0.2),
                            "C" = c(0.1, 0.6, 0.3)
  )

  return (occur_fraction)
}

#' Get mu_1, mu_2 functions, and score_vals objects for a given context.
#' @param setting setting identified as an integer 1,2,3
#' @param scenario scenario name as a string "A", "B", "C"
#' @param k number of points along the score decay axis
#' @return A list that contains mu_1, mu_2, score_vals
#'
GetMuAndScore <- function(setting, scenario, k)
{
  all_score_values = rep(0, k)

  if(1 == setting)
  {
    mu_1 <- function(t) -1 + 2 * t + 2 * t^2

    mu_2 <- switch(scenario,
                   "A" = function(t) -2.5 + exp(t * 2),
                   "B" = function(t) -0.5 + exp(t * 2),
                   "C" = function(t) -2.5 + exp(t * 2)
    )
    score_front <- switch(scenario,
                        "A" = c(1, 1/2, 1/4),
                        "B" = c(1, 1/2, 1/4),
                        "C" = c(50, 25, 5)
    )
  } else if(2 == setting)
  {
    mu_1 <- function(t) 4 * t^2 - 1.2

    mu_2 <- function(t) 4 * t^2 - 3.5

    score_front <- c(1, 1/2, 1/4)
  } else if(3 == setting)
  {
    mu_1 <- function(t) -2.2 + 4 * t^2

    mu_2 <- function(t) -7 + 6 * t^2

    score_front <- c(1, 1/4, 1/16)
  }

  for(idx in 1:length(score_front))
  {
    all_score_values[idx] <- score_front[idx]
  }

  return(list("mu_1" = mu_1, "mu_2" = mu_2, "score_vals" = all_score_values))
}


#' Generate cluster data for a given scenario
#' @param num_indvs number of individuals
#' @param timeseries_length length of time-series as an integer
#' @param k description?? number of psi functions
#' @param mu_1 description??
#' @param mu_2 description??
#' @param score_vals description??
#'
GenerateClusterDataScenario <- function(num_indvs,
                                 timeseries_length,
                                 k = 3,
                                 mu_1,
                                 mu_2,
                                 score_vals)
{
  timestamps01 <- seq(from = 0.0001, to = 1, length=timeseries_length)

  # noise octaves
  cat("octave", num_indvs, k, num_indvs * k, "\n")
  scores_standard <- matrix(rnorm(num_indvs * k), ncol = k)
  scores <- scores_standard %*% diag(sqrt(score_vals))

  #
  BIG_mu <- c(mu_1(timestamps01), mu_2(timestamps01))
  BIG_phi <- PsiFunc(k, timestamps01)

  Z <- BIG_phi %*% t(scores) + BIG_mu
  Z1 <- Z[1:timeseries_length, ]
  Z2 <- Z[1:timeseries_length + timeseries_length, ]
  expZ1 <- exp(Z1)
  expZ2 <- exp(Z2)
  denom <- 1 + expZ1 + expZ2
  p1 <- expZ1 / denom
  p2 <- expZ2 / denom
  p3 <- 1 / denom

  # vectorize for future work!!!
  return(list(Z1 = Z1, Z2 = Z2,
              p1 = p1, p2 = p2, p3 = p3,
              MEAN = BIG_mu, PHI = BIG_phi, MFPC = scores))
}

#' Psi function
#'
PsiFunc <- function(klen, timestamps01)
{
  psi_k1 <- sapply(c(1:klen), function(i) sin((2 * i + 1) * pi * timestamps01))
  psi_k2 <- sapply(c(1:klen), function(i) cos(2 * i * pi * timestamps01))
  return(rbind(psi_k1, psi_k2))
}

# EXECUTION:

result_par <- ClusterSimulation(100, 15, "A", 5)

print(result)

# }) # profvis end

# parallel::stopCluster(cl = my.cluster)
