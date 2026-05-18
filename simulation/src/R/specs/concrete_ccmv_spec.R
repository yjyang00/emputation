# src/R/specs/concrete_ccmv_spec.R

concrete_ccmv_spec = function(d) {
  stopifnot(d == 9)
  
  mkpat = function(miss, d) {
    p = rep("1", d); p[miss] = "0"; paste0(p, collapse = "")
  }
  
  patterns = c(
    mkpat(c(1, 2, 3, 4), d), 
    mkpat(c(5, 6, 7, 8), d), 
    mkpat(c(9, 1, 2, 3), d),
    mkpat(c(4, 5, 6, 7), d),
    mkpat(c(8, 9, 1, 2), d),
    mkpat(c(3, 4, 5, 6), d),
    mkpat(c(7, 8, 9, 1), d),
    mkpat(c(2, 3, 4, 5), d),
    mkpat(c(6, 7, 8, 9), d),
    mkpat(c(1, 3, 5, 7), d),
    mkpat(c(2, 4, 6, 8, 9), d) # The 5-column balancer
  )
  
  coeffs = lapply(patterns, function(p_str) {
    p_vec = as.numeric(strsplit(p_str, "")[[1]])
    num_observed = sum(p_vec == 1)
    c(-4.0, rep(1, num_observed)) 
  })
  
  list(patterns = patterns, coeffs = coeffs)
}