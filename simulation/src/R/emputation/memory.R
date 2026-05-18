estimate_mb = function(N_terms, M, d, hidden_dim, num_layer,
                        bytes_f = 4, bytes_bool = 1, backward_mult = 2.5) {
  B = N_terms * M
  in_dim  = 2 * d
  out_dim = d
  
  # data tensors: X_base_pre, r_batch_pre, yt_r_pre, yhat_all, yhat_r, E
  mb_data = 6 * B * d * bytes_f / 1024^2
  
  # xt: (B, 2d)
  mb_xt = B * in_dim * bytes_f / 1024^2
  
  # r_obs_mask_pre, T_r_mask_pre
  mb_masks  = 2 * B * d * bytes_bool / 1024^2
  
  # hidden activations stored for backprop per layer
  mb_hidden = backward_mult * max(num_layer - 1, 1) * B * hidden_dim * bytes_f / 1024^2
  
  # cdist: output (N_terms, M, M) + internal intermediate (N_terms, M, M, d)
  mb_cdist = N_terms * M^2 * (1 + d) * bytes_f / 1024^2
  
  # optimizer: Adam stores 2 moment estimates per parameter
  n_params = in_dim * hidden_dim +                        # first layer weights
    hidden_dim +                                  # first layer bias
    (num_layer - 1) * (hidden_dim * hidden_dim + hidden_dim) +  # hidden layers
    hidden_dim * out_dim + out_dim                # output layer
  mb_optimizer = 2 * n_params * bytes_f / 1024^2
  
  total = mb_data + mb_xt + mb_masks + mb_hidden + mb_cdist + mb_optimizer
  
  list(
    total_mb  = round(total, 1),
    breakdown = round(c(
      data      = mb_data,
      xt        = mb_xt,
      masks     = mb_masks,
      hidden    = mb_hidden,
      cdist     = mb_cdist,
      optimizer = mb_optimizer
    ), 1)
  )
}