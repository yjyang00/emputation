# src/R/specs/wine_ccmv_spec.R

wine_ccmv_spec = function(d) {
  stopifnot(d == 12)
  
  mkpat = function(miss, d) {
    p = rep("1", d); p[miss] = "0"; paste0(p, collapse = "")
  }
  
  patterns = c(
    mkpat(c(1, 2, 3, 4, 5, 6), d),
    mkpat(c(7, 8, 9, 10, 11, 12), d),
    mkpat(c(1, 3, 5, 7, 9, 11), d),
    mkpat(c(2, 4, 6, 8, 10, 12), d),
    mkpat(c(1, 4, 5, 8, 9, 12), d),
    mkpat(c(2, 3, 6, 7, 10, 11), d),
    mkpat(c(1, 2, 7, 8, 3, 9), d),
    mkpat(c(4, 5, 10, 11, 6, 12), d),
    mkpat(c(1, 2, 3, 4, 5), d), mkpat(c(6, 7, 8, 9, 10), d),
    mkpat(c(11, 12, 1, 2, 3), d), mkpat(c(4, 5, 6, 7, 8), d),
    mkpat(c(9, 10, 11, 12, 2), d), mkpat(c(3, 4, 5, 6, 7), d),
    mkpat(c(8, 9, 10, 11, 1), d), mkpat(c(12, 2, 3, 4, 7), d),
    mkpat(c(5, 6, 8, 9, 12), d), mkpat(c(1, 3, 7, 10, 11), d),
    mkpat(c(2, 4, 5, 8, 9), d), mkpat(c(6, 10, 11, 12, 1), d)
  )
  
  coeffs = lapply(patterns, function(p_str) {
    p_vec = as.numeric(strsplit(p_str, "")[[1]])
    num_observed = sum(p_vec == 1)
    c(-4.0, rep(1, num_observed)) 
  })
  
  list(patterns = patterns, coeffs = coeffs)
}