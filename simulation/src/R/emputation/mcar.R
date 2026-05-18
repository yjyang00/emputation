generate_mcar = function(data, p, cc_prop = 0.2) {
  n_rows = nrow(data)
  n_cols = ncol(data)
  
  n_cc = round(cc_prop * n_rows)
  cc_rows = sample(1:n_rows, n_cc, replace = FALSE)
  mcar_rows = setdiff(1:n_rows, cc_rows)
  
  n_mcar = length(mcar_rows)
  mask_matrix = matrix(FALSE, nrow = n_rows, ncol = n_cols)
  mcar_mask = matrix(runif(n_mcar * n_cols) < p, nrow = n_mcar)
  
  all_na_rows = which(rowSums(mcar_mask) == n_cols)
  if (length(all_na_rows) > 0) {
    safe_cols = sample(1:n_cols, length(all_na_rows), replace = TRUE)
    mcar_mask[cbind(all_na_rows, safe_cols)] = FALSE
  }
  
  mask_matrix[mcar_rows, ] = mcar_mask
  
  dat_NA = data
  dat_NA[mask_matrix] = NA
  return(dat_NA)
}