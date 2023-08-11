# Load global settings
source("~/.Rprofile")

# Cores for parallel processing
Sys.setenv(OMP_NUM_THREADS = parallel::detectCores() - 1)
options_default <- options()
#options_default <- options(options_default)
options(width = 120, repr.matrix.max.cols = 150, repr.matrix.max.rows = 100)

# Load local settings

## Libraries
library(chatgpt)
library(R2OpenBUGS)

## Working directory
setwd("/home/pal_bjartan/Backup/PhD/SEM-test-model/Lee2007")
