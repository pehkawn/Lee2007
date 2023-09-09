data {
    int<lower=1> N; // number of observations
    int<lower=1> P; // number of outcome variables
    matrix[N, P] y; // outcome data
}

parameters {
    matrix[N, 2] xi; // latent variables
    vector[N] eta; // latent variable
    vector<lower=0>[2] psi; // measurement error precisions
    real<lower=0> psd; // standard deviation of eta
    matrix<lower=-1, upper=1>[2, 2] phi; // covariance matrix
    vector[9] alp; // intercepts
    real lam[6]; // loadings
    real gam[2]; // coefficients for nu[i]
}

transformed parameters {
    matrix[2, 2] phx; // inverse covariance matrix
    vector[N] nu; // intermediate variable
    vector[N] dthat; // intermediate variable

    phx = inverse(phi);

    for (i in 1:N) {
        nu[i] = gam[1] * xi[i, 1] + gam[2] * xi[i, 2];
        dthat[i] = eta[i] - nu[i];
    }
}

model {
    // Measurement equation model
    for (i in 1:N) {
        vector[P] mu;

        mu[1] = eta[i] + alp[1];
        mu[2] = lam[1] * eta[i] + alp[2];
        mu[3] = lam[2] * eta[i] + alp[3];
        mu[4] = xi[i, 1] + alp[4];
        mu[5] = lam[3] * xi[i, 1] + alp[5];
        mu[6] = lam[4] * xi[i, 1] + alp[6];
        mu[7] = xi[i, 2] + alp[7];
        mu[8] = lam[5] * xi[i, 2] + alp[8];
        mu[9] = lam[6] * xi[i, 2] + alp[9];

        y[i] ~ multi_normal(mu, diag_matrix(psi));
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
    phi[1:2, 1:2] ~ wishart(5, R[1:2, 1:2]);
}

