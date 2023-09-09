data {
  int<lower=0> N; // number of observations
  int<lower=0> P; // number of variables
  matrix[N, P] y; // observed data
  vector[2] u = [0, 0]; // prior mean for xi
  matrix[2, 2] R = [ [1.0, 0.0], [0.0, 1.0] ]; // prior covariance matrix for xi
}

transformed data {
   
}

parameters {
    vector[P] alp; // intercepts
    vector[N] eta; // latent variable
    matrix[N, 2] xi; // latent variables
    real lam[6]; // loadings
    real gam[2]; // coefficients for nu[i]
    vector<lower=0>[P] psi; // precisions of y
    real<lower=0> psd; // precision of eta
    matrix[2, 2] phi;

}

transformed parameters {
   vector[P] sgm = 1/psi;
   real sgd = 1/psd;
   matrix[2, 2] phx = inverse(phi);
}


model {
    // Measurement equation model
    for (i in 1:N) {
        vector[P] mu;
        vector[P] ephat;
        
        mu[1] = eta[i] + alp[1];
        mu[2] = lam[1] * eta[i] + alp[2];
        mu[3] = lam[2] * eta[i] + alp[3];
        mu[4] = xi[i, 1] + alp[4];
        mu[5] = lam[3] * xi[i, 1] + alp[5];
        mu[6] = lam[4] * xi[i, 1] + alp[6];
        mu[7] = xi[i, 2] + alp[7];
        mu[8] = lam[5] * xi[i, 2] + alp[8];
        mu[9] = lam[6] * xi[i, 2] + alp[9];

        y[i] ~ normal(mu[i], psi);
        ephat[i] = y[i] - mu[i]
    }

    // Structural equation model
    xi ~ multi_normal(rep_vector(0, 2), phi);
    eta ~ normal(nu, psd);

    // Priors
    for (j in 1:9)
        alp[j] ~ normal(0, 1);
    
    lam[1] ~ normal(0.8, psi[2]);
    lam[2] ~ normal(0.8, psi[3]);
    lam[3] ~ normal(0.8, psi[5]);
    lam[4] ~ normal(0.8, psi[6]);
    lam[5] ~ normal(0.8, psi[8]);
    lam[6] ~ normal(0.8, psi[9]);
    for (j in 1:2)
        gam[j] ~ normal(0.5, psd);

    // Priors om precisions
    for (j in 1:P) {
        psi[j] ~ gamma(9, 4);
    }
    psd ~ gamma(9, 4);
    // phi ~ wishart(5, diag_matrix(rep_vector(1, 2)));
    phi ~ wishart(5, R[1:2, 1:2]);
}