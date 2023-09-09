## ---------------------------------------------------------------------------------------------------------------------
source(".Rprofile")


## ---------------------------------------------------------------------------------------------------------------------
model  <- function() {
    for(i in 1:N){
        #measurement equation model
        for(j in 1:P){
            y[i,j] ~ dnorm(mu[i,j],psi[j])
            ephat[i,j] <- y[i,j] - mu[i,j]
        }
        mu[i,1] <- eta[i]+alp[1]
        mu[i,2] <- lam[1] * eta[i] + alp[2]
        mu[i,3] <- lam[2] * eta[i] + alp[3]
        mu[i,4] <- xi[i,1] + alp[4]
        mu[i,5] <- lam[3] * xi[i,1] + alp[5]
        mu[i,6] <- lam[4] * xi[i,1] + alp[6]
        mu[i,7] <- xi[i,2] + alp[7]
        mu[i,8] <- lam[5] * xi[i,2] + alp[8]
        mu[i,9] <- lam[6] * xi[i,2] + alp[9]

        #structural equation model
        xi[i,1:2] ~ dmnorm(u[1:2], phi[1:2,1:2])
        eta[i] ~ dnorm(nu[i], psd)
        nu[i] <- gam[1] * xi[i,1] + gam[2] * xi[i,2]
        dthat[i] <- eta[i] - nu[i]
    } #end of i

    #priors on intercepts
    for(j in 1:9){alp[j]~dnorm(0.0, 1.0)}

    #priors on loadings and coefficients
    lam[1] ~ dnorm(0.8, psi[2])
    lam[2] ~ dnorm(0.8, psi[3])
    lam[3] ~ dnorm(0.8, psi[5])
    lam[4] ~ dnorm(0.8, psi[6])
    lam[5] ~ dnorm(0.8, psi[8])
    lam[6] ~ dnorm(0.8, psi[9])
    for(j in 1:2){ gam[j] ~ dnorm(0.5, psd) }
    
    #priors on precisions
    for(j in 1:P){
        psi[j] ~ dgamma(9.0, 4.0)
        sgm[j] <- 1/psi[j]
    }
    psd ~ dgamma(9.0, 4.0)
    sgd <- 1/psd
    phi[1:2,1:2] ~ dwish(R[1:2,1:2], 5)
    phx[1:2,1:2] <- inverse(phi[1:2, 1:2])

} #end of model

write.model(model, con = "./Chapter4/ch4-R2OpenBUGS-model.txt")


## ---------------------------------------------------------------------------------------------------------------------
model <- paste0(getwd(), "/Chapter4/ch4-R2OpenBUGS-model.txt")


## ---------------------------------------------------------------------------------------------------------------------
# Read in dataset as unnamed matrix
YO.dat <- read.csv("./Chapter4/ch4-WinBUGS-data.dat", header = FALSE, skip = 2)[,1:9] %>% 
as.matrix()  %>% 
unname()

# Save data as list in the following format
data <- list(
   N = 300,
   P = 9,
   u = c(0,0),
   y = YO.dat,
   R = structure(
      .Data=c( 1.0, 0.0,
               0.0, 1.0),
      .Dim = c(2,2))
)


## ---------------------------------------------------------------------------------------------------------------------
# Three different initial values
inits <- function() {
   list(alp = c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      lam = c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      psi = c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
      psd = 1.0,
      gam = c(1.0, 1.0),
      phi = structure(
         .Data = c(  0.2, 0.1,
                     0.1, 0.9),
         .Dim = c(2,2)
      )
   )

   list(alp = c(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0),
      lam = c(-1.0, 0.0, 0.3, 0.6, 0.9, 1.0),
      psi = c(1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
      psd = 1.5,
      gam = c(-1.0,-1.0),
      phi = structure(
         .Data = c(  0.5, 0.2,
                     0.2, 0.6),
         .Dim = c(2,2)
      )
   )

   list(alp = c(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
      lam = c(1.0, 0.3, 0.4, 0.5, 1.0, -1.0),
      psi = c(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
      psd = 0.3,
      gam = c(2.0, 2.0),
      phi = structure(
         .Data = c(  0.9, 0.3,
                     0.3, 0.8),
         .Dim = c(2,2)
      )
   )
}


## ---------------------------------------------------------------------------------------------------------------------
param <- c("alp", "lam", "psi", "phi", "gam", "xi")


## ---------------------------------------------------------------------------------------------------------------------
n.iter <- 5000


## ---------------------------------------------------------------------------------------------------------------------
n.burnin  <- 2000


## ---------------------------------------------------------------------------------------------------------------------
model.out <- bugs(
    data, 
    inits, 
    param[-6], 
    model.file = model, 
    n.iter, 
    n.burnin = n.burnin, 
    codaPkg = TRUE,
    working.directory = paste0(getwd(), "/Chapter4/bugs-output")
)

codaobject <- read.bugs(model.out)
plot(codaobject)
save.image()


## ---------------------------------------------------------------------------------------------------------------------
model.out <- bugs(
    data, 
    inits, 
    param, 
    model.file = model, 
    2*n.iter, 
    n.burnin = n.burnin, 
    codaPkg = FALSE,  # Get bugs object
    working.directory = paste0(getwd(), "/Chapter4/bugs-output")
)
model.out
save.image()

