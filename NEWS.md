# catfda 0.1.0

* Initial CRAN release.
* GAM-based estimation of latent Gaussian processes from categorical functional
  data: binomial (logit), probit, and multinomial link functions.
* Parallel estimation via `foreach`/`doRNG` for all three model families.
* `generate_categ_func_data()` for simulating categorical functional data from
  a multivariate latent Gaussian process.
* Methods described in Champon et al. (2026)
  <doi:10.1080/01621459.2026.2672226>.
