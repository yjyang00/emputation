ccmv = function(data, patterns = NULL, coeffs = NULL, standardize = TRUE) {
  dat_NA = data
  if(standardize){
    data = as.data.frame(scale(data))
  }
  n = nrow(data)
  d = ncol(data)
  
  if (is.null(patterns)) {
    complete = paste(rep("1", d), collapse = "")
    single_missing = sapply(1:d, function(i) {
      p = rep("1", d)
      p[i] = "0"
      paste(p, collapse = "")
    })
    patterns = c(complete, single_missing)
  }
  
  patterns = unique(c(paste(rep("1", d), collapse = ""), patterns))
  missing_patterns = patterns[patterns != paste(rep("1", d), collapse = "")]
  
  # selection odds
  if (is.null(coeffs)) {
    coeffs = lapply(missing_patterns, function(p) {
      runif(sum(strsplit(p, "")[[1]] == "1") + 1, -0.5, 0.5)
    })
  }
  
  odds_matrix = matrix(0, nrow = n, ncol = length(missing_patterns))
  
  for (j in 1:length(missing_patterns)) {
    p_str = missing_patterns[j]
    p_vec = as.numeric(strsplit(p_str, "")[[1]])
    obs_idx = which(p_vec == 1)
    X_obs = as.matrix(cbind(1, data[, obs_idx]))
    odds_matrix[, j] = exp(X_obs %*% coeffs[[j]])
  }
  
  prob_complete = 1 / (1 + rowSums(odds_matrix))
  prob_matrix = cbind(prob_complete, prob_complete * odds_matrix)
  colnames(prob_matrix) = c(patterns[1], missing_patterns)
  
  final_data = data
  assigned_patterns = sapply(1:n, function(i) {
    sample(colnames(prob_matrix), size = 1, prob = prob_matrix[i, ])
  })
  
  for (i in 1:n) {
    p_vec = as.numeric(strsplit(assigned_patterns[i], "")[[1]])
    final_data[i, p_vec == 0] = NA
  }
  
  dat_NA[is.na(final_data)] = NA
  return(dat_NA)
}


