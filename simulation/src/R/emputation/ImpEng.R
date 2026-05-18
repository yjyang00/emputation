emputation = function(dat, mechanism, M=2, hidden_dim=500, num_layer=3, 
                      batch_norm=TRUE, num_epochs=1000, lr=10^(-4), beta=1, 
                      standardize=TRUE, silent=FALSE, force_full_batch = TRUE,
                      batch_size = NULL, print_every_nepoch = 100, 
                      verbose_memory=FALSE, mem_threshold_mb = 3000){
  
  if (is.data.frame(dat)) {
    if (any(sapply(dat, is.factor))) warning("Data frame contains factor variables. Mapping to numeric values. Dummy variables would need to be created explicitly by the user.")
    dat = dftomat(dat)
  }
  
  if (is.vector(dat) && !is.numeric(dat)) dat = as.numeric(dat)
  if (is.vector(dat) && is.numeric(dat)) dat = matrix(dat, ncol = 1)
  
  muX = apply(dat, 2, function(col) mean(col, na.rm = TRUE))
  sddX = apply(dat, 2, function(col) sd(col, na.rm = TRUE))
  if(any(sddX<=0)){
    warning("Some variables have zero variance on observed data -- results might be unreliable")
    sddX = pmax(sddX, 10^3)
  }
  
  dat_std = dat
  if(standardize){
    dat_std  = sweep(sweep(dat,2,muX,FUN="-"),2,sddX,FUN="/")
  }
  
  if(mechanism == 'mcar'){
    R = apply(!is.na(dat), 1, function(x) paste0(as.numeric(x), collapse=""))
    Rmat = 1 * (!is.na(dat))
    Rset = unique(R)
    Rls = all_patterns_less_than(Rset) # look-up list
    
    R_type_index = match(R, names(Rls)) 
    
    unique_r_strs = unique(unlist(Rls)) # lookup table
    r_pattern_table = do.call(rbind, lapply(unique_r_strs, function(s) as.integer(strsplit(s, "")[[1]])))
    r_pattern_index = setNames(seq_along(unique_r_strs), unique_r_strs)
    
    N_terms_max = sum(sapply(R_type_index, function(idx) length(Rls[[idx]])))
    i_indices   = integer(N_terms_max)
    r_idx_vec   = integer(N_terms_max) 
    Tr_counts   = integer(N_terms_max)
    k = 1
    for(i in 1:nrow(dat)){
      Ri_vec = as.integer(Rmat[i, ])
      rls = Rls[[R_type_index[i]]]
      if(length(rls) == 0) next
      
      for(r_str in rls){
        r_vec = r_pattern_table[r_pattern_index[[r_str]], ]
        i_indices[k] = i
        r_idx_vec[k] = r_pattern_index[[r_str]]
        Tr_counts[k] = sum(Ri_vec - r_vec)
        k = k + 1
      }
    }
    
    batch_metadata = list(
      N_terms         = k - 1,
      i_indices       = i_indices,
      r_pattern_table = r_pattern_table,  
      r_idx_vec       = r_idx_vec,
      Tr_counts       = Tr_counts,
      d               = ncol(dat),
      n               = nrow(dat)
    )
  }
  if(mechanism == 'ccmv'){
    R = apply(!is.na(dat), 1, function(x) paste0(as.numeric(x), collapse=""))
    Rmat = 1 * (!is.na(dat))
    Rset = unique(R)
    Rcc = paste0(rep(1, ncol(dat)), collapse = "")
    Rset = Rset[! Rset == Rcc ] 
    r_pattern_table = do.call(rbind, lapply(Rset, function(s) as.integer(strsplit(s, "")[[1]])))
    r_pattern_index = setNames(seq_along(Rset), Rset)
    
    N_cc        = sum(rowSums(Rmat) == ncol(dat))
    N_terms_max = N_cc * length(Rset)
    i_indices   = integer(N_terms_max)
    r_idx_vec   = integer(N_terms_max)
    Tr_counts   = integer(N_terms_max)
    k = 1
    
    for(i in 1:nrow(dat)){
      if(paste0(as.integer(Rmat[i, ]), collapse="") != Rcc) next 
      
      for(r_str in Rset){
        r_vec = r_pattern_table[r_pattern_index[[r_str]], ]
        i_indices[k] = i
        r_idx_vec[k] = r_pattern_index[[r_str]]
        Tr_counts[k] = sum(1 - r_vec) #|\bar{r}|
        k = k + 1
      }
    }
    
    # metadata
    batch_metadata = list(
      N_terms = k - 1,
      i_indices = i_indices,
      r_pattern_table = r_pattern_table,
      r_idx_vec = r_idx_vec,
      Tr_counts = Tr_counts,
      d = ncol(dat),
      n = nrow(dat)
    )
  }
  message("Metadata preparation complete. Starting emputation...")
  emp = emputationfit(dat=dat_std, mechanism = mechanism, batch_metadata=batch_metadata, 
                      Rmat=Rmat, M=M, hidden_dim=hidden_dim, num_layer=num_layer, 
                      batch_norm=batch_norm, num_epochs=num_epochs, lr=lr, beta=beta, 
                      batch_size=batch_size, silent=silent, 
                      print_every_nepoch = print_every_nepoch, force_full_batch=force_full_batch,
                      mem_threshold_mb = mem_threshold_mb, verbose_memory=verbose_memory)
  emputor = list(emputor = emp$emputor, lossvec= emp$lossvec, mechanism = mechanism, 
                 M=M, Rmat=Rmat, batch_metadata=batch_metadata, muX=muX, sddX=sddX, 
                 standardize=standardize, hidden_dim=hidden_dim, num_layer=num_layer, 
                 batch_norm=batch_norm, num_epochs=num_epochs, lr=lr, model = emp$model,
                 batch_size=batch_size)
  class(emputor) = "emputation"
  return(emputor)
}

