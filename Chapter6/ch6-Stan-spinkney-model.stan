data{
	int<lower=0> N; // number of observations
    int<lower=1> P; // number of variables
    array[N, P] int<lower=1, upper=5> z; // observed data
	array[P] ordered[6] thd; // Thresholds
}

transformed data {
   	cov_matrix[4] R = diag_matrix(rep_vector(8.0, 4)); // prior covariance matrix for xi
	vector[4] u = rep_vector(0, 4); // prior mean for xi
    array[N] vector[P] L;
    array[N] vector[P] U;
    for(i in 1:N){
		for(j in 1:P){
			L[i,j] = thd[j, z[i, j]];
            U[i,j] = thd[j, z[i, j] + 1];
		}
    }
}

parameters {
	array[21] real lam; // pattern coefficients
	row_vector[4] gam; // coefficients
	vector<lower=0>[P] psi; // precisions of y
	real<lower=0> psd; // precision of eta
	cholesky_factor_corr[4] L_phi;
  array[N] vector<lower=L, upper=U>[P] y;
  array[N] vector[4] xi; // latent variables
  vector[N] eta; // latent variables
	
}
model {
	// Local variables
    vector[P] mu;
    psi ~ inv_gamma(10,8); //priors on precisions
    
	//priors on loadings and coefficients
    lam[1] ~ normal(0.8, 4.0 * psi[2]);
    lam[2] ~ normal(0.8, 4.0 * psi[4]);
    lam[3] ~ normal(0.8, 4.0 * psi[5]);
    lam[4] ~ normal(0.8, 4.0 * psi[6]);
    lam[5] ~ normal(0.8, 4.0 * psi[7]);
    lam[6] ~ normal(0.8, 4.0 * psi[8]);
    lam[7] ~ normal(0.8, 4.0 * psi[9]);
    lam[8] ~ normal(0.8, 4.0 * psi[11]);
    lam[9] ~ normal(0.8, 4.0 * psi[12]);
    lam[10] ~ normal(0.8, 4.0 * psi[13]);
    lam[11] ~ normal(0.8, 4.0 * psi[14]);
    lam[12] ~ normal(0.8, 4.0 * psi[15]);
    lam[13] ~ normal(0.8, 4.0 * psi[17]);
    lam[14] ~ normal(0.8, 4.0 * psi[18]);
    lam[15] ~ normal(0.8, 4.0 * psi[20]);
    lam[16] ~ normal(0.8, 4.0 * psi[21]);
    lam[17] ~ normal(0.8, 4.0 * psi[22]);
    lam[18] ~ normal(0.8, 4.0 * psi[23]);
    lam[19] ~ normal(0.8, 4.0 * psi[24]);
    lam[20] ~ normal(0.8, 4.0 * psi[25]);
    lam[21] ~ normal(0.8, 4.0 * psi[26]);
    
    psd ~ inv_gamma(10,8); //priors on precisions
    gam ~ normal(0.6, 4 * psd);
    
    L_phi ~ lkj_corr_cholesky(1);
   // L_phi ~ inv_wishart_cholesky(30, R); //priors on precisions
    xi ~ multi_normal_cholesky(u, L_phi);
    
    for(i in 1:N){
        // structural equation model
        real nu = gam * xi[i];
        // real eta;
        eta[i] ~ normal(nu, psd);
        real dthat = eta[i] - nu;
        

        //measurement equation model
        
        mu[1] = eta[i];
        mu[2] = lam[1] * eta[i];
        mu[3] = xi[i, 1];
        mu[4] = lam[2] * xi[i, 1];
        mu[5] = lam[3] * xi[i, 1];
        mu[6] = lam[4] * xi[i, 1];
        mu[7] = lam[5] * xi[i, 1];
        mu[8] = lam[6] * xi[i, 1];
        mu[9] = lam[7] * xi[i, 1];
        mu[10] = xi[i, 2];
        mu[11] = lam[8] * xi[i, 2];
        mu[12] = lam[9] * xi[i, 2];
        mu[13] = lam[10] * xi[i, 2];
        mu[14] = lam[11] * xi[i, 2];
        mu[15] = lam[12] * xi[i, 2];
        mu[16] = xi[i, 3];
        mu[17] = lam[13] * xi[i, 3];
        mu[18] = lam[14] * xi[i, 3];
        mu[19] = xi[i, 4];
        mu[20] = lam[15] * xi[i, 4];
        mu[21] = lam[16] * xi[i, 4];
        mu[22] = lam[17] * xi[i, 4];
        mu[23] = lam[18] * xi[i, 4];
        mu[24] = lam[19] * xi[i, 4];
        mu[25] = lam[20] * xi[i, 4];
        mu[26] = lam[21] * xi[i, 4];

        y[i] ~ normal(mu, psi);	

	} // end of i
} //end of model