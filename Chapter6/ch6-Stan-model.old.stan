data{
	int<lower=0> N; // number of observations
    int<lower=1> P; // number of variables
    array[N, P] int<lower=1, upper=5> z; // observed data
    array[P, 6] real thd; // Thresholds
}

transformed data {
   	cov_matrix[4] R = diag_matrix(rep_vector(8.0, 4)); // prior covariance matrix for xi
	vector[4] u = rep_vector(0,4); // prior mean for xi
    array[N, P] real L; // Lower truncation point of y
    array[N, P] real U; // Upper truncation point of y
    for(i in 1:N){
		for(j in 1:P){
			L[i,j] = thd[j, z[i, j]]; // Lower truncation point assigned values from thd[j, 1:5] based on value of z[i,j]
            U[i,j] = thd[j, z[i, j] + 1]; // Lower truncation point assigned values from thd[j, 2:6] based on value of z[i,j]
		}
    }
}

parameters {
	array[21] real lam; // pattern coefficients
	vector[N] eta; // latent variables
	array[N] vector[4] xi; // latent variables
	row_vector[4] gam; // coefficients
	vector<lower=0>[P] psi; // precisions of y
	real<lower=0> psd; // precision of eta
	cov_matrix[4] phi;
    array[N, P] real<lower=L, upper=U> y; // Not sure if this is a correct assumption, but I chose a 2-D array of reals, as I figured this might let me assign constraints for each value of y.
	
}

transformed parameters {
    vector[P] sgm = 1/psi;
    real sgd = 1/psd;
    matrix[4, 4] phx = inverse(phi);
}

model {
	// Local variables
    array[N, P] real mu;
    array[N, P] real ephat;
    vector[N] nu;
    vector[N] dthat;
	array[21] real var_lam;
	real var_gam;

	//priors on precisions
    psi ~ gamma(10,8);
    psd ~ gamma(10,8);
    phi ~ wishart(30, R);

	//priors on loadings and coefficients
    var_lam[1] = 4.0 * psi[2];
    var_lam[2] = 4.0 * psi[4];
    var_lam[3] = 4.0 * psi[5];
    var_lam[4] = 4.0 * psi[6];
    var_lam[5] = 4.0 * psi[7];
    var_lam[6] = 4.0 * psi[8];
    var_lam[7] = 4.0 * psi[9];
    var_lam[8] = 4.0 * psi[11];
    var_lam[9] = 4.0 * psi[12];
    var_lam[10] = 4.0 * psi[13];
    var_lam[11] = 4.0 * psi[14];
    var_lam[12] = 4.0 * psi[15];
    var_lam[13] = 4.0 * psi[17];
    var_lam[14] = 4.0 * psi[18];
    var_lam[15] = 4.0 * psi[20];
    var_lam[16] = 4.0 * psi[21];
    var_lam[17] = 4.0 * psi[22];
    var_lam[18] = 4.0 * psi[23];
    var_lam[19] = 4.0 * psi[24];
    var_lam[20] = 4.0 * psi[25];
    var_lam[21] = 4.0 * psi[26];
    
    lam ~ normal(0.8,var_lam);

    var_gam = 4.0 * psd;

    gam ~ normal(0.6,var_gam);

    for(i in 1:N){
        //measurement equation model
        
        mu[i,1] = eta[i];
        mu[i,2] = lam[1] * eta[i];
        mu[i,3] = xi[i,1];
        mu[i,4] = lam[2] * xi[i,1];
        mu[i,5] = lam[3] * xi[i,1];
        mu[i,6] = lam[4] * xi[i,1];
        mu[i,7] = lam[5] * xi[i,1];
        mu[i,8] = lam[6] * xi[i,1];
        mu[i,9] = lam[7] * xi[i,1];
        mu[i,10] = xi[i,2];
        mu[i,11] = lam[8] * xi[i,2];
        mu[i,12] = lam[9] * xi[i,2];
        mu[i,13] = lam[10] * xi[i,2];
        mu[i,14] = lam[11] * xi[i,2];
        mu[i,15] = lam[12] * xi[i,2];
        mu[i,16] = xi[i,3];
        mu[i,17] = lam[13] * xi[i,3];
        mu[i,18] = lam[14] * xi[i,3];
        mu[i,19] = xi[i,4];
        mu[i,20] = lam[15] * xi[i,4];
        mu[i,21] = lam[16] * xi[i,4];
        mu[i,22] = lam[17] * xi[i,4];
        mu[i,23] = lam[18] * xi[i,4];
        mu[i,24] = lam[19] * xi[i,4];
        mu[i,25] = lam[20] * xi[i,4];
        mu[i,26] = lam[21] * xi[i,4];

		for(j in 1:P){
            ephat[i, j] = y[i, j] - mu[i, j]; 
            y[i,j] ~ normal(mu[i,j], psi[j])T[L[i, j], U[i, j]];
        }
		

        // structural equation model

		nu[i] = gam * xi[i];
	} // end of i
        
	dthat = eta - nu;

	xi ~ multi_normal(u, phi);
	eta ~ normal(nu, psd);

} //end of model