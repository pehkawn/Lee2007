data{
	int<lower=0> N; // number of observations
    int<lower=1> P; // number of variables
    array[N, P] int<lower=1, upper=5> z; // observed data
}

transformed data {
   	cov_matrix[4] R = diag_matrix(rep_vector(8.0, 4)); // prior covariance matrix for xi
	vector[4] u = rep_vector(0,4); // prior mean for xi
}

parameters {
	array[21] real lam; // pattern coefficients
	vector[N] eta; // latent variables
	array[N] vector[4] xi; // latent variables
	row_vector[4] gam; // coefficients
	vector<lower=0>[P] psi; // precisions of y
	real<lower=0> psd; // precision of eta
	cov_matrix[4] phi;
    array[P] ordered[4] thd; // Thresholds
	
}

transformed parameters { 
    
}

model {
	// Local variables
    vector[P] mu;
    // vector[N] nu;
    // vector[N] dthat;

    psi ~ gamma(10,8); //priors on covariance of lam
    vector[P] sgm = 1/psi; //priors on precision of psi

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
    
     // structural equation model

    psd ~ gamma(10,8); //prior on covariance of gam
    real sgd = 1/psd; //prior on precision of psd
    gam ~ normal(0.6, 4 * psd);
    
    // phi ~ wishart(30, R); //priors on covariance matrix of xi
    // phi ~ inv_wishart_cholesky(30, R); //priors on cholesky factor of covariance
    phi ~ wishart_cholesky(30, R); //priors on cholesky factor of covariance
    matrix[4, 4] phx = inverse(phi); //priors on precisions of phi
    
    // xi ~ multi_normal(u, phi);
    xi ~ multi_normal_cholesky(u, phi);
    
    // eta ~ normal(gam * xi, psd);

    for(i in 1:N){

       
        // real nu;
        // nu = gam * xi[i];
        eta[i] ~ normal(gam * xi[i], psd);
        // real dthat = eta[i] - nu;
        

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

		z[i] ~ ordered_logistic(mu, thd);
        // for(j in 1:P){
        //     // y[i,j] ~ normal(mu[i,j], psi[j]) * step(thd[j, z[i,j]] - y[i,j]);
		// 	z[i,j] ~ ordered_logistic(mu[i,j], thd[j]);
        // }
        
		
        
	} // end of i

} //end of model

