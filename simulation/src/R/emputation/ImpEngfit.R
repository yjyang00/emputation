emputationfit = function(dat, mechanism, batch_metadata, Rmat, 
                         M=2, hidden_dim=500, num_layer=3,
                         batch_norm=TRUE, num_epochs=500, lr=10^(-4), 
                         beta=1, batch_size = NULL, silent=FALSE, verbose_memory=FALSE,
                         print_every_nepoch = 100, 
                         force_full_batch = FALSE, 
                         mem_threshold_mb = 500){
  
  N_terms = batch_metadata$N_terms
  n = batch_metadata$n 
  d = batch_metadata$d
  
  dat_t = torch_tensor(dat, dtype = torch_float())
  Tr_counts_all_t = torch_tensor(batch_metadata$Tr_counts)

  i_indices_all = batch_metadata$i_indices 
  r_pattern_table_t = torch_tensor(batch_metadata$r_pattern_table)
  r_vectors_all_t = r_pattern_table_t[batch_metadata$r_idx_vec, ]

  if(mechanism == 'mcar'){
    Rmat_all_t = torch_tensor(Rmat)
  }
  
  full_batch = is.null(batch_size)
  cat(full_batch)
  est = estimate_mb(N_terms, M, d, hidden_dim, num_layer)
  estimated_mb = est$total_mb
  if (verbose_memory){
    message(sprintf("Estimated memory usage: %.0f MB", est$total_mb))
  }
  
  if(full_batch && force_full_batch && estimated_mb > mem_threshold_mb){
    warning("Estimated memory usage is large. This may cause memory issues.")
  }
  
  if(full_batch && !force_full_batch && estimated_mb > mem_threshold_mb){
    message(sprintf(
      "Estimated memory usage %.0f MB > threshold %.0f MB. Switching to minibatch (batch_size=128). Set force_full_batch=TRUE to override.",
      estimated_mb, mem_threshold_mb
    ))
    full_batch = FALSE
    batch_size = 128
    eff_batch_size = 128
  }
  
  if(full_batch){
    X_base_pre = dat_t[i_indices_all, ]$repeat_interleave(M, dim=1)   # (N_terms*M, d)
    r_batch_pre = r_vectors_all_t$repeat_interleave(M, dim=1)           # (N_terms*M, d)
    r_obs_mask_pre = (r_batch_pre == 1)     
    
    if(mechanism == 'mcar'){
      Rmat_terms = Rmat[i_indices_all, ] # (N_terms, d)
      T_r_mask_mat = (Rmat_terms - batch_metadata$r_pattern_table[batch_metadata$r_idx_vec, ]) == 1 # Ri-r
      T_r_mask_pre = torch_tensor(T_r_mask_mat, dtype=torch_bool())$repeat_interleave(M, dim=1)
    } else {
      # ccmv
      T_r_mask_mat = batch_metadata$r_pattern_table[batch_metadata$r_idx_vec, ] == 0 # \bar{r}
      T_r_mask_pre = torch_tensor(T_r_mask_mat, dtype=torch_bool())$repeat_interleave(M, dim=1)
    }
    
    yt_r_pre = torch_where(T_r_mask_pre, X_base_pre, torch_tensor(0, dtype=torch_float()))$view(list(N_terms, M, d))
    
  } else {
    eff_batch_size = min(batch_size, n)
    i_to_terms_named = split(seq_len(N_terms), i_indices_all)
    i_to_terms = vector("list", n)
    i_to_terms[as.integer(names(i_to_terms_named))] = i_to_terms_named
  }
  
  in_dim  = 2 * d
  out_dim = d
  if(num_layer <= 2){
    if(!batch_norm){
      model = nn_sequential(nn_linear(in_dim, hidden_dim), nn_relu(), nn_linear(hidden_dim, out_dim))
    } else {
      model = nn_sequential(nn_linear(in_dim, hidden_dim), nn_relu(), nn_batch_norm1d(hidden_dim), nn_linear(hidden_dim, out_dim))
    }
  } else {
    if(!batch_norm){
      hid = nn_sequential(nn_linear(hidden_dim, hidden_dim), nn_relu())
      if(num_layer > 3) for(lay in 3:num_layer) hid = nn_sequential(hid, nn_sequential(nn_linear(hidden_dim, hidden_dim), nn_relu()))
      model = nn_sequential(nn_sequential(nn_linear(in_dim, hidden_dim), nn_relu()), hid, nn_linear(hidden_dim, out_dim))
    } else {
      hid = nn_sequential(nn_linear(hidden_dim, hidden_dim), nn_relu(), nn_batch_norm1d(hidden_dim))
      if(num_layer > 3) for(lay in 3:num_layer) hid = nn_sequential(hid, nn_sequential(nn_linear(hidden_dim, hidden_dim), nn_relu(), nn_batch_norm1d(hidden_dim)))
      model = nn_sequential(nn_sequential(nn_linear(in_dim, hidden_dim), nn_relu(), nn_batch_norm1d(hidden_dim)), hid, nn_linear(hidden_dim, out_dim))
    }
  }
  model$train()
  optimizer = optim_adam(model$parameters, lr=lr)
  
  # training loop
  lossvec = matrix(nrow=num_epochs, ncol=3)
  colnames(lossvec) = c("energy-loss", "Term 1", "Term 2")
  row_indices = 1:n
  num_batches = if(full_batch) 1 else ceiling(n / eff_batch_size)
  
  for(epoch in 1:num_epochs){
    epoch_losses = c(0, 0, 0)
    
    if(full_batch){
      optimizer$zero_grad()
      
      E = torch_randn(list(N_terms*M, d))
      X_input = torch_where(r_obs_mask_pre, X_base_pre, E)
      xt = torch_cat(list(X_input, r_batch_pre), dim=2)   # (N_terms*M, 2d)
      
      yhat_all = model(xt)
      yhat_r = (yhat_all * T_r_mask_pre)$view(list(N_terms, M, d))
      
      # ====== Term 1: mean over M of ||yhat - y|| ======
      d1 = torch_pow(torch_norm(yt_r_pre - yhat_r, p=2, dim=3), beta)
      s1_term = torch_mean(d1, dim=2)   # (N_terms)
      
      # ====== Term 2: all-pairs within yhat_r, scaled by M/(M-1) ======
      pair_dist = torch_pow(torch_cdist(yhat_r, yhat_r, p=2), beta)   # (N_terms, M, M)
      s2_term = torch_mean(pair_dist, dim=list(2,3)) * (M / (M-1))   # (N_terms)
      
      loss = torch_sum((s1_term - 0.5 * s2_term) / Tr_counts_all_t) / n
      
      loss$backward()
      optimizer$step()
      
      epoch_losses = c(
        loss$item(),
        (torch_sum(s1_term / Tr_counts_all_t) / n)$item(),
        (torch_sum(s2_term / Tr_counts_all_t) / n)$item()
      )
    } else {
      perm_rows = sample(row_indices, replace=FALSE)
      
      for(b in 1:num_batches){
        optimizer$zero_grad()
        
        curr_rows = perm_rows[((b-1)*eff_batch_size + 1):min(b*eff_batch_size, n)]
        B_rows = length(curr_rows)
        
        curr_idx = unlist(i_to_terms[curr_rows], use.names=FALSE)
        if(length(curr_idx) == 0) next
        B_terms = length(curr_idx)
        
        batch_i = i_indices_all[curr_idx]
        X_base = dat_t[batch_i, ]$repeat_interleave(M, dim=1)
        r_batch = r_vectors_all_t[curr_idx, ]$repeat_interleave(M, dim=1)
        
        if(mechanism == 'mcar'){
          Ri_batch = Rmat_all_t[batch_i, ]$repeat_interleave(M, dim=1)
          T_r_mask = (Ri_batch == 1) & (r_batch == 0)
        } else {
          T_r_mask = (r_batch == 0)
        }
        
        Y_true_batch = torch_where(T_r_mask, X_base, torch_tensor(0, dtype=torch_float()))
        yt_r = Y_true_batch$view(list(B_terms, M, d))
        
        E = torch_randn(list(B_terms*M, d))
        X_input = torch_where(r_batch == 1, X_base, E)
        xt = torch_cat(list(X_input, r_batch), dim=2)
        
        yhat_all = model(xt)
        yhat_r = (yhat_all * T_r_mask)$view(list(B_terms, M, d))
        
        d1 = torch_pow(torch_norm(yt_r - yhat_r, p=2, dim=3), beta)
        s1_term = torch_mean(d1, dim=2)
        
        pair_dist = torch_pow(torch_cdist(yhat_r, yhat_r, p=2), beta)
        s2_term = torch_mean(pair_dist, dim=list(2,3)) * (M / (M-1))
        
        Tr_counts = Tr_counts_all_t[curr_idx]
        loss = torch_sum((s1_term - 0.5 * s2_term) / Tr_counts) / B_rows
        
        loss$backward()
        optimizer$step()
        epoch_losses = epoch_losses + c(
          loss$item(),
          (torch_sum(s1_term / Tr_counts) / B_rows)$item(),
          (torch_sum(s2_term / Tr_counts) / B_rows)$item()
        )
      }
    }
    
    # save
    lossvec[epoch, ] = epoch_losses / num_batches
    if(!silent){
      cat(sprintf("\rEpoch [%4d/%4d] | loss: %.4f | T1: %.4f | T2: %.4f",
                  epoch, num_epochs,
                  lossvec[epoch,1], lossvec[epoch,2], lossvec[epoch,3]))
      flush.console()
      if(epoch == 1 || epoch %% print_every_nepoch == 0 || epoch == num_epochs) cat("\n")
    }
  }
  
  if(batch_norm) model$train(mode=FALSE)
  
  multiemputor = function(x_new, m=5){
    x_new = as.matrix(x_new)
    n = nrow(x_new); d = ncol(x_new)
    out = array(NA, dim=c(m, n, d))
    r_mat = 1 * (!is.na(x_new))
    r_tensor = torch_tensor(r_mat, dtype=torch_float())
    
    x_base = x_new
    x_base[is.na(x_base)] = 0
    x_base_tensor = torch_tensor(x_base, dtype=torch_float())
    model$eval()
    
    with_no_grad({
      for(t in 1:m){
        eps = torch_randn(list(n, d))
        eps_masked = eps * (1 - r_tensor)
        x_batch = x_base_tensor + eps_masked
        xt = torch_cat(list(x_batch, r_tensor), dim=2)
        emp_all = model(xt)
        X_imp_m = x_base_tensor + (emp_all * (1 - r_tensor))
        out[t, , ] = as.matrix(X_imp_m)
      }
    })
    
    return(out)
  }
  
  return(list(model=model, emputor=multiemputor, lossvec=lossvec))
}


