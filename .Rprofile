# Load global settings
source("~/.Rprofile")

# Cores for parallel processing
Sys.setenv(OMP_NUM_THREADS = parallel::detectCores() - 1)
options_default <- options()
#options_default <- options(options_default)
options(width = 120, repr.matrix.max.cols = 150, repr.matrix.max.rows = 100)

# Load local settings

if(file.exists(".RData")) load(".RData")

## Libraries
# library(chatgpt)
library(R2OpenBUGS)
library(rstan)

## Working directory
setwd("/home/pal_bjartan/Backup/PhD/SEM-test-model/Lee2007")

save.image()
