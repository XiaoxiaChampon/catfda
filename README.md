# catfda

R package for **categorical functional data analysis**. Implements the methods
described in:

> Champon X, Staicu A-M, Weishampel A, Jayalath C, Rand W (2026).
> "Clustering Social Media Users Using Categorical-Valued Functional Data Analysis."
> *Journal of the American Statistical Association*.
> <https://doi.org/10.1080/01621459.2026.2672226>

## Overview

Categorical functional data arise when each subject is observed as a sequence
of categorical values over a dense time grid (e.g., daily tweet-type activity).
`catfda` estimates the latent Gaussian processes underlying such data using
GAM-based approaches, providing the inputs needed for downstream FPCA-based
clustering.

**Core functions:**

| Function | Description |
|---|---|
| `estimate_categ_func_data()` | Main dispatcher: estimate latent Z and probability curves |
| `estimate_categ_func_data_multinomial()` | Multinomial GAM estimation |
| `estimate_categ_func_data_probit()` | Probit/adaptive estimation |
| `estimate_categ_func_data_binomial_parallel()` | Parallel binomial estimation |
| `get_x_from_w()` | Convert categorical matrix to one-hot array |
| `generate_categ_func_data()` | Simulate categorical functional data |

## Installation

```r
# Install from GitHub
remotes::install_github("XiaoxiaChampon/catfda")
```

## Usage

The example below reproduces the core workflow from the paper: simulate three
groups of social media users with distinct tweeting patterns, estimate the
latent probability curves, extract MFPCA scores, and cluster.

```r
library(catfda)
library(doParallel)   # parallel backend
library(refund)       # for fpca.face()
library(NbClust)      # for cluster count selection
library(fossil)       # for rand.index(), adj.rand.index()

# ---- 1. Simulate 3 clusters of users (T=250, n=100) ----
set.seed(123)
T  <- 250
n1 <- 50; n2 <- 30; n3 <- 20
tt <- seq(0, 1, length.out = T)

sim_cluster <- function(n_users, mu1_fn, mu2_fn, seed_offset = 0) {
  W <- matrix(NA, nrow = T, ncol = n_users)
  for (i in seq_len(n_users)) {
    set.seed(123 + seed_offset + i)
    z1    <- mu1_fn(tt) + rnorm(T, 0, 0.3)
    z2    <- mu2_fn(tt) + rnorm(T, 0, 0.3)
    denom <- 1 + exp(z1) + exp(z2)
    p     <- cbind(exp(z1)/denom, exp(z2)/denom, 1/denom)
    W[, i] <- apply(p, 1, function(pr) sample(1:3, 1, prob = pr))
  }
  W
}

W1 <- sim_cluster(n1, function(t)  3.8 + 4*t^2 - 5, function(t) 1.5 + 4*t^2 - 5)
W2 <- sim_cluster(n2, function(t)  0.97 + 6*t^2 - 8, function(t) 0.50 + 4*t^2 - 6, seed_offset = 100)
W3 <- sim_cluster(n3, function(t) -sin(2*pi*t) + 1,  function(t) 3*t - 1.5,         seed_offset = 200)

W           <- cbind(W1, W2, W3)
true_labels <- c(rep(1, n1), rep(2, n2), rep(3, n3))

# ---- 2. Estimate latent Z and probability curves ----
cl  <- makeCluster(2); registerDoParallel(cl)
est <- estimate_categ_func_data("multinomial", tt, W, n_basis = 25)
stopCluster(cl)
# Returns: z1_est, z2_est (T x n), p1_est, p2_est, p3_est (T x n)

# ---- 3. Extract MFPCA scores ----
out1   <- fpca.face(Y = t(est$z1_est), argvals = tt, pve = 0.95)
out2   <- fpca.face(Y = t(est$z2_est), argvals = tt, pve = 0.95)
scores <- cbind(out1$scores, out2$scores)

# ---- 4. Cluster with k-means (silhouette, 2-5 clusters) ----
fit    <- NbClust(data = scores, distance = "euclidean",
                  min.nc = 2, max.nc = 5, method = "kmeans", index = "silhouette")
labels <- fit$Best.partition

# ---- 5. Evaluate ----
rand.index(true_labels, labels)
adj.rand.index(true_labels, labels)
```

## Citation

```r
citation("catfda")
```

## License

MIT © 2026 Xiaoxia Champon, Ana-Maria Staicu, Chathura Jayalath

