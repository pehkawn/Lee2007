data{
	array[50] int<lower=0> N; // number of observations per group
    int<lower=1> P; // number of variables
    array[N, P] int<lower=1, upper=5> z; // observed data
	array[P] ordered[4] thd; // Thresholds
}

transformed data {
   	cov_matrix[4] R = diag_matrix(rep_vector(8.0, 4)); // prior covariance matrix for xi
	vector[4] u = rep_vector(0,4); // prior mean for xi
}

model
{
    for (g in 1:50) {
        for (i in 1:N[g]) {
            for (j in 1:9) {
                y[kk[g] + i, j] ~ dnorm(u[kk[g] + i, j], psi[j])
                ephat[kk[g] + i, j] = y[kk[g] + i, j] - u[kk[g] + 
                  i, j]
            }
            u[kk[g] + i, 1] = mu[1] + pi[g, 1] + eta[g, i]
            u[kk[g] + i, 2] = mu[2] + lb[1] * pi[g, 1] + lw[1] * 
                eta[g, i]
            u[kk[g] + i, 3] = mu[3] + lb[2] * pi[g, 1] + lw[2] * 
                eta[g, i]
            u[kk[g] + i, 4] = mu[4] + pi[g, 2] + xi[g, i, 1]
            u[kk[g] + i, 5] = mu[5] + lb[3] * pi[g, 2] + lw[3] * 
                xi[g, i, 1]
            u[kk[g] + i, 6] = mu[6] + lb[4] * pi[g, 2] + lw[4] * 
                xi[g, i, 1]
            u[kk[g] + i, 7] = mu[7] + pi[g, 3] + xi[g, i, 2]
            u[kk[g] + i, 8] = mu[8] + lb[5] * pi[g, 3] + lw[5] * 
                xi[g, i, 2]
            u[kk[g] + i, 9] = mu[9] + lb[6] * pi[g, 3] + lw[6] * 
                xi[g, i, 2]
            xi[g, i, 1:2] ~ dmnorm(ux[1:2], phi[1:2, 1:2])
            eta[g, i] ~ dnorm(nu[g, i], psd)
            nu[g, i] = gam[1] * xi[g, i, 1] + gam[2] * xi[g, 
                i, 2] + gam[3] * xi[g, i, 1] * xi[g, i, 2]
            dthat[g, i] = eta[g, i] - nu[g, i]
        }
        pi[g, 1:3] ~ dmnorm(uu[1:3], phip[1:3, 1:3])
    }
    uu[1] = 0.00000E+00
    uu[2] = 0.00000E+00
    uu[3] = 0.00000E+00
    ux[1] = 0.00000E+00
    ux[2] = 0.00000E+00
    mu[1] ~ dnorm(4.248, 4)
    mu[2] ~ dnorm(4.668, 4)
    mu[3] ~ dnorm(4.56, 4)
    mu[5] ~ dnorm(3.161, 4)
    mu[6] ~ dnorm(3.445, 4)
    mu[7] ~ dnorm(0.526, 4)
    mu[8] ~ dnorm(0.375, 4)
    mu[9] ~ dnorm(0.596, 4)
    var.bw[1] = 4 * psi[2]
    var.bw[2] = 4 * psi[3]
    var.bw[3] = 4 * psi[5]
    var.bw[4] = 4 * psi[6]
    var.bw[5] = 4 * psi[8]
    var.bw[6] = 4 * psi[9]
    lb[1] ~ dnorm(1.096, var.bw[1])
    lb[2] ~ dnorm(0.861, var.bw[2])
    lb[3] ~ dnorm(0.59, var.bw[3])
    lb[4] ~ dnorm(1.47, var.bw[4])
    lb[5] ~ dnorm(0.787, var.bw[5])
    lb[6] ~ dnorm(0.574, var.bw[6])
    lw[1] ~ dnorm(0.825, var.bw[1])
    lw[2] ~ dnorm(0.813, var.bw[2])
    lw[3] ~ dnorm(0.951, var.bw[3])
    lw[4] ~ dnorm(0.692, var.bw[4])
    lw[5] ~ dnorm(0.986, var.bw[5])
    lw[6] ~ dnorm(0.8, var.bw[6])
    var.gam = 4 * psd
    gam[1] ~ dnorm(0.577, var.gam)
    gam[2] ~ dnorm(1.712, var.gam)
    gam[3] ~ dnorm(-0.571, var.gam)
    for (j in 1:9) {
        psi[j] ~ dgamma(10, 4)
        ivpsi[j] = 1/psi[j]
    }
    psd ~ dgamma(10, 4)
    ivpsd = 1/psd
    phi[1:2, 1:2] ~ dwish(R0[1:2, 1:2], 5)
    phx[1:2, 1:2] = inverse(phi[1:2, 1:2])
    phip[1:3, 1:3] ~ dwish(R1[1:3, 1:3], 5)
    php[1:3, 1:3] = inverse(phip[1:3, 1:3])
}
