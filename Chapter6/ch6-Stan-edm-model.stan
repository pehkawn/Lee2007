data{
  int<lower=0> N; // number of observations
  int<lower=1> P; // number of variables
  array[N, P] int<lower=1, upper=5> z; // observed data
}
transformed data {
  cov_matrix[4] R = diag_matrix(rep_vector(1.0, 4)); // prior covariance matrix for xi
  vector[4] u = rep_vector(0, 4); // mean for xi
  array[N, P] int ones; // to use bernoulli likelihood

  for (i in 1:N) for (j in 1:P) ones[i,j] = 1;
}
parameters {
  vector[21] lam; // pattern coefficients
  row_vector[4] gam; // coefficients
  real<lower=0> psd; // precision of eta
  cov_matrix[4] phi;
  array[N] vector[4] xi; // latent variables
  vector[N] eta_std; // latent variables
  array[P] ordered[4] thd;
}
transformed parameters {
  array[N] vector[P] L;
  array[N] vector[P] U;
  array[N] vector[P] mu;
  matrix[4, 4] phx = cholesky_decompose(phi);
  vector[N] eta;
  real sgd = pow(psd, -.5);
  
  for(i in 1:N){
    eta[i] = sgd * eta_std[i] + gam * xi[i];
    for(j in 1:P){
      if (z[i,j] == 1) {
        L[i,j] = negative_infinity();
      } else {
        L[i,j] = thd[j, z[i, j] - 1];
      }
      if (z[i,j] == 5) {
        U[i,j] = positive_infinity();
      } else {
        U[i,j] = thd[j, z[i, j]];
      }
    }
    
    // measurement equation model, matrices would make this more concise:
    mu[i,1] = eta[i];
    mu[i,2] = lam[1] * eta[i];
    mu[i,3] = xi[i, 1];
    mu[i,4] = lam[2] * xi[i, 1];
    mu[i,5] = lam[3] * xi[i, 1];
    mu[i,6] = lam[4] * xi[i, 1];
    mu[i,7] = lam[5] * xi[i, 1];
    mu[i,8] = lam[6] * xi[i, 1];
    mu[i,9] = lam[7] * xi[i, 1];
    mu[i,10] = xi[i, 2];
    mu[i,11] = lam[8] * xi[i, 2];
    mu[i,12] = lam[9] * xi[i, 2];
    mu[i,13] = lam[10] * xi[i, 2];
    mu[i,14] = lam[11] * xi[i, 2];
    mu[i,15] = lam[12] * xi[i, 2];
    mu[i,16] = xi[i, 3];
    mu[i,17] = lam[13] * xi[i, 3];
    mu[i,18] = lam[14] * xi[i, 3];
    mu[i,19] = xi[i, 4];
    mu[i,20] = lam[15] * xi[i, 4];
    mu[i,21] = lam[16] * xi[i, 4];
    mu[i,22] = lam[17] * xi[i, 4];
    mu[i,23] = lam[18] * xi[i, 4];
    mu[i,24] = lam[19] * xi[i, 4];
    mu[i,25] = lam[20] * xi[i, 4];
    mu[i,26] = lam[21] * xi[i, 4];
  }
}

model {
  for(i in 1:N){
    xi[i] ~ multi_normal_cholesky(u, phx);

    eta_std[i] ~ std_normal();

    for(j in 1:P){
      // Phi_approx() leads to not finite gradient:
      ones[i,j] ~ bernoulli(Phi(U[i,j] - mu[i,j]) - Phi(L[i,j] - mu[i,j]));
    }
  } // end of i
    
  //priors on loadings and coefficients
  lam ~ normal(1, .4); // (what you call psi is fixed to 1 for these models)
  psd ~ gamma(1, .5); // prior on eta precision
  gam ~ normal(0.6, 1);
  phi ~ inv_wishart(5, R); // prior on xi covariance matrix
}
generated quantities {
  array[N] real dthat;

  for (i in 1:N) {
    dthat[i] = eta[i] - gam * xi[i];
  }
}
