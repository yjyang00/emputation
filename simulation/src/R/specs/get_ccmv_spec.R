# src/R/specs/get_ccmv_spec.R

source("src/R/specs/ccpp_ccmv_spec.R")
source("src/R/specs/wine_ccmv_spec.R")
source("src/R/specs/concrete_ccmv_spec.R")

get_ccmv_spec = function(dataset, d) {
  dataset = tolower(dataset)
  
  if (dataset == "ccpp")     return(ccpp_ccmv_spec(d))
  if (dataset == "wine")     return(wine_ccmv_spec(d))
  if (dataset == "concrete") return(concrete_ccmv_spec(d))
  
  stop("No CCMV spec defined for dataset: ", dataset)
}
