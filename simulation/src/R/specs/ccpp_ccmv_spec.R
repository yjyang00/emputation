# src/R/specs/ccpp_ccmv_spec.R

ccpp_ccmv_spec = function(d) {
  stopifnot(d == 5)
  
  mkpat = function(miss, d) {
    p = rep("1", d); p[miss] = "0"; paste0(p, collapse = "")
  }
  
  patterns = c(
    mkpat(c(1, 2), d), mkpat(c(2, 3), d), mkpat(c(3, 4), d), mkpat(c(4, 5), d), mkpat(c(5, 1), d),
    mkpat(c(1, 3), d), mkpat(c(2, 4), d), mkpat(c(3, 5), d), mkpat(c(4, 1), d), mkpat(c(5, 2), d),
    
    mkpat(c(1), d), mkpat(c(2), d), mkpat(c(3), d), mkpat(c(4), d), mkpat(c(5), d), # 5 Singles
    mkpat(c(1, 2, 3), d), mkpat(c(2, 3, 4), d), mkpat(c(3, 4, 5), d), # 5 Triples
    mkpat(c(4, 5, 1), d), mkpat(c(5, 1, 2), d)
  )
  
  coeffs = lapply(patterns, function(p_str) {
    p_vec = as.numeric(strsplit(p_str, "")[[1]])
    num_observed = sum(p_vec == 1)
    c(-4.0, rep(1, num_observed)) 
  })
  
  list(patterns = patterns, coeffs = coeffs)
}