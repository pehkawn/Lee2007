data {
  int<lower=0> N; // number of observations
  int<lower=0> P; // number of variables
  matrix[N, P] y; // observed data
  vector[2] u = c(0, 0); // prior mean for xi
  matrix[2, 2] R = matrix(c(1.0, 0.0, 0.0, 1.0), nrow=2, ncol=2); // prior covariance matrix for xi
}

parameters {
  vector[P] alp; // intercepts
  vector[P] lam; // loadings and coefficients
  vector[2] gam; // structural equation model coefficients
  real<lower=0> psd; // precision of eta
  vector<lower=0>[P] psi; // precisions of y
}

transformed parameters {
  matrix[N, 2] xi; // latent variables
  vector[N] nu; // means of eta
  vector[N] dthat; // deviations of eta from nu

  for (i in 1:N) {
    xi[i] = mvn_draw(u, phi);
    nu[i] = gam[1] * xi[i, 1] + gam[2] * xi[i, 2];
    dthat[i] = eta[i] - nu[i];
  }
}

model {
  // priors
  for (j in 1:P) {
    psi[j] ~ gamma(9.0, 4.0);
    lam[j] ~ normal(0.8, psi[2]);
  }
  for (j in 1:2) {
    gam[j] ~ normal(0.5, psd);
  }
  psd ~ gamma(9.0, 4.0);
  phi ~ wishart(R, 5);

  // likelihood
  for (i in 1:N) {
    for (j in 1:P) {
      y[i, j] ~ normal(mu[i, j], psi[j]);
    }
  }
}