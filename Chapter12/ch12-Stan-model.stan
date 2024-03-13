data{
	int<lower=0> N; // number of observations
    int<lower=1> P; // number of variables
    cov_matrix[2] RR; // prior covariance matrix for xi
	vector[2] u; // prior mean for xi
    array[N] vector[P] y; // observed data
    array[N, P] int<lower=0, upper=1> R;
	}

transformed data {
    vector[8] mu_vu = [-0.145, -0.086, 0.012, 0.004, -0.143, -0.036, 0.029, 0.143]';
    vector[9] mu_b = [-2.798, 0.041, -0.281, 0.365, -0.264, -0.524, -0.275, -0.061, 0.327]';
    
    array[5] int psi_ind = {2, 4, 5, 7, 8};
    vector[5] mu_lam = [0.490, 0.188, 0.194, 0.537, 0.226]';
    vector[3] mu_gam = [-0.072, -0.005, 0.206]';
}

parameters {
	vector<lower=0>[P] psi; // precisions of y
    real<lower=0> psd; // precision of eta
    cov_matrix[2] phi; //priors on covariance matrix of xi
    vector[P] vu;
    vector[9] b;
    vector[5] lam; // pattern coefficients
	vector[3] gam; // coefficients
	matrix[N, 2] xi; // latent variables
    vector[N] xxi;
    vector[N] eta; // latent variables
    
	
}

transformed parameters {
    vector[P] v = 1/psi; //priors on precision of psi
    real vd = 1/psd; //prior on precision of psd
    matrix[4, 4] phx = inverse(phi); //priors on precisions of phi
    
    // structural equation
    vector[N] nu = xi * gam[:2] + gam[3] * square(xi[:, 1]);
    vector[N] dthat = xxi - nu;
    
    // measurement models
    matrix[N, P] mu;
    mu[:, 1] = vu[1] + xxi;
    mu[:, 2] = vu[2] + lam[1] * xxi;
    mu[:, 3] = vu[3] + xi[:, 1];
    mu[:, 4:5] = vu[4:5] + lam[2:3] * xi[:, 1];
    mu[:, 6] = vu[6] + xi[:, 2];
    mu[:, 7:8] = vu[7:8] + lam[4:5] * xi[:, 2];

    vector[N] ephat = y - mu;

    // put all the parameters' results into bb
    vector[37] bb;
    bb[1:8] = vu;
    bb[9:13] = lam;
    bb[14:21] = v;
    bb[22:24] = gam;
    bb[25] = vd;
    bb[26] = phx[1, 1] ;
    bb[27] = phx[1, 2] ;
    bb[28] = phx[2, 2];
    bb[29:37] = b;
}

model {
    // priors on precisions
    psi ~ gamma(10.0, 4.0); //priors on covariance of lam
    psd ~ gamma(10.0, 4.0); //prior on covariance of gam    
    phi ~ wishart(2, RR); //priors on covariance matrix of xi
    

    // priors on loadings and coefficients
    
    vu ~ normal(mu_vu, rep_vector(4.0, 8));
    b ~ normal(mu_b, rep_vector(4.0, 9));
       
    
    lam ~ normal(mu_lam, 4.0 * psi[psi_ind]);
        
    gam ~ normal(mu_gam, 4.0 * psd);
    
    // structural equation
    xi ~ multi_normal(u, phi);
    xxi ~ normal(nu, psd);

    // missingness mechanism model
    R ~ bernoulli_logit(b[1] + b[2:9] * y);
    
    // measurement models
    y ~ normal(mu, psi);	
    
    // end of model
}
