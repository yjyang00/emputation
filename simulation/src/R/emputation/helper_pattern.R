patterns_less_than = function(Ri, set){
  Ri_vec = as.numeric(strsplit(Ri, "")[[1]])
  Rset_mat = do.call(rbind, strsplit(set, "")) |> apply(2, as.numeric)
  cond1 = apply(Rset_mat, 1, function(r) all(r <= Ri_vec)) 
  cond2 = apply(Rset_mat, 1, function(r) any(r < Ri_vec)) # strictly smaller
  
  return(set[cond1 & cond2])
}

all_patterns_less_than = function(set) {
  
  out = lapply(1:length(set), function(i) {
    patterns_less_than(set[i], set)
  })
  
  names(out) = set
  
  return(out)
}
