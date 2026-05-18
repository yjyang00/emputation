# src/R/emputation/mar.R

generate_mar = function(data, target_rate = 0.4, cc_prop = 0.2) {
  data = as.data.frame(data)
  N    = nrow(data)
  d    = ncol(data)
  n_cc   = round(cc_prop * N)
  cc_idx = sample(1:N, n_cc)
  amp_idx = setdiff(1:N, cc_idx)
  amp_data = data[amp_idx, ]
  N_amp = nrow(amp_data)
  
  x = apply(amp_data, 2, function(col) {
    r = range(col, na.rm = TRUE)
    if (r[2] == r[1]) return(rep(0, N_amp))
    (col - r[1]) / (r[2] - r[1])
  })
  
  w = runif(d)
  b = runif(d)
  m = matrix(1, nrow = N_amp, ncol = d)
  adjusted_target = target_rate / (1 - cc_prop)

  for (i in 1:d) {
    
    if (i == 1) {
      prob_missing = rep(adjusted_target, N_amp)
      
    } else {
      score = numeric(N_amp)
      for (j in 1:(i-1)) {
        score = score + w[j] * m[, j] * x[, j] + b[j] * (1 - m[, j])
      }
      exp_score = exp(-score)
      Z         = sum(exp_score)
      
      prob_missing = adjusted_target * N_amp * exp_score / Z
      prob_missing = pmin(pmax(prob_missing, 0), 1)
    }
    
    m[, i] = rbinom(N_amp, 1, 1 - prob_missing)
  }
  
  amp_NA = amp_data
  for (i in 1:d) {
    amp_NA[m[, i] == 0, i] = NA
  }
  
  dat_NA = data
  dat_NA[amp_idx, ] = amp_NA
  all_na_rows = which(rowSums(is.na(dat_NA)) == d)
  if (length(all_na_rows) > 0) {
    for (r in all_na_rows) {
      safe_col = sample(1:d, 1)
      dat_NA[r, safe_col] = data[r, safe_col]
    }
  }
  
  return(dat_NA = dat_NA)
}

