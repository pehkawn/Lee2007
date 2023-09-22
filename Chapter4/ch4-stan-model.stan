data {
    int<lower=0> N; // number of observations
    int<lower=1> P; // number of variables
    array[N] vector[P] y; // observed data
}

transformed data {
    vector[2] u = rep_vector(0, 2); // prior mean for xi
    cov_matrix[2] R = identity_matrix(2); // prior covariance matrix for xi
}


parameters {
    
    vector[P] alp; // intercepts
    array[6] real lam; // loadings
    vector[N] eta; // latent variable
    array[N] vector[2] xi; // latent variables
    // matrix[N, 2] xi; // latent variables
    row_vector[2] gam; // coefficients for nu[i]
    vector<lower=0>[P] psi; // precisions of y
    real<lower=0> psd; // precision of eta
    cov_matrix[2] phi;

}

transformed parameters {
    vector[P] sgm = 1/psi;
    real sgd = 1/psd;
    matrix[2, 2] phx = inverse(phi);   
}


model {
    // Local variables
    array[N] vector[P] mu;
    array[N] vector[P] ephat;
    vector[N] nu;
    vector[N] dthat; 
    

    // Priors on intercepts
    alp ~ normal(0, 1);

    // Priors om precisions
    psi ~ gamma(9, 4);
    psd ~ gamma(9, 4); // precision of eta
    phi ~ wishart(5, R); //diag_matrix(rep_vector(1, 2)));
    
    // Priors on loadings and coefficients
    lam[1] ~ normal(0.8, psi[2]);
    lam[2] ~ normal(0.8, psi[3]);
    lam[3] ~ normal(0.8, psi[5]);
    lam[4] ~ normal(0.8, psi[6]);
    lam[5] ~ normal(0.8, psi[8]);
    lam[6] ~ normal(0.8, psi[9]);
    
    gam ~ normal(0.5, psd);

    // Measurement equation model
    
    for (i in 1:N) {
        
        mu[i,1] = eta[i] + alp[1];
        mu[i,2] = lam[1] * eta[i] + alp[2];
        mu[i,3] = lam[2] * eta[i] + alp[3];
        mu[i,4] = xi[i, 1] + alp[4];
        mu[i,5] = lam[3] * xi[i, 1] + alp[5];
        mu[i,6] = lam[4] * xi[i, 1] + alp[6];
        mu[i,7] = xi[i, 2] + alp[7];
        mu[i,8] = lam[5] * xi[i, 2] + alp[8];
        mu[i,9] = lam[6] * xi[i, 2] + alp[9];

        ephat[i] = y[i] - mu[i];
        y[i] ~ normal(mu[i], psi);
    
    // Structural equation model
        
        nu[i] = gam * xi[i];
    }

    dthat = eta - nu;

    xi ~ multi_normal(u, phi);
    eta ~ normal(nu, psd);
}

