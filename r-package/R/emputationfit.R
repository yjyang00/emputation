#' Fit the emputation model
#'
#' Builds the feed-forward neural network, trains it by minimizing the empirical
#' emputation risk, and returns a closure that draws multiple imputations.
#'
#' The masking/term bookkeeping in `batch_metadata` is mechanism-specific and
#' is prepared by [emputation()]. 
#'
#' @param dat A standardized numeric matrix, `n` * `d`.
#' @param mechanism One of `"mcar"`, `"ccmv"`, `"tree"`.
#' @param batch_metadata A list prepared by [emputation()].
#' @param Rmat An `n` * `d` 0/1 observed-indicator matrix.
#' @param B,hidden_dim,num_layer,batch_norm,num_epochs,lr,beta,batch_size,
#'   silent,print_every_nepoch See [emputation()].
#'
#' @return A list with `model` (the trained `torch` `nn_module`), `emputor`
#'   (a function `function(x_new, m)` drawing `m` imputations of `x_new`),
#'   and `lossvec` (an `num_epochs` * 3 matrix of training losses).
#' @keywords internal
#' @noRd
emputationfit <- function(dat, mechanism, batch_metadata, Rmat,
                           B = 2, hidden_dim = 500, num_layer = 3,
                           batch_norm = TRUE, num_epochs = 500, lr = 10^(-4),
                           beta = 1, batch_size = NULL, silent = FALSE,
                           print_every_nepoch = 100) {

  N_terms <- batch_metadata$N_terms
  n <- batch_metadata$n
  d <- batch_metadata$d

  dat_t <- torch::torch_tensor(dat, dtype = torch::torch_float())
  Tr_counts_all_t <- torch::torch_tensor(batch_metadata$Tr_counts, dtype = torch::torch_float())

  i_indices_all <- batch_metadata$i_indices
  r_pattern_table_t <- torch::torch_tensor(batch_metadata$r_pattern_table, dtype = torch::torch_float())
  r_vectors_all_t <- r_pattern_table_t[batch_metadata$r_idx_vec, ]

  if (mechanism == "mcar") {
    Rmat_terms <- Rmat[i_indices_all, , drop = FALSE]
    T_r_mask_mat <- (Rmat_terms - batch_metadata$r_pattern_table[batch_metadata$r_idx_vec, , drop = FALSE]) == 1 # Ri - r
  } else if (mechanism == "tree") {
    T_r_mask_mat <- batch_metadata$T_mask_table[batch_metadata$r_idx_vec, , drop = FALSE] == 1 # PA(r) - r
  } else { # ccmv
    T_r_mask_mat <- batch_metadata$r_pattern_table[batch_metadata$r_idx_vec, , drop = FALSE] == 0 # \bar{r}
  }
  T_r_mask_all_t <- torch::torch_tensor(T_r_mask_mat, dtype = torch::torch_bool())

  i_to_terms_named <- split(seq_len(N_terms), i_indices_all)
  i_to_terms <- vector("list", n)
  i_to_terms[as.integer(names(i_to_terms_named))] <- i_to_terms_named

  eff_batch_size <- if (is.null(batch_size)) n else min(batch_size, n)
  num_batches <- ceiling(n / eff_batch_size)

  in_dim <- 2 * d
  out_dim <- d
  if (num_layer <= 2) {
    if (!batch_norm) {
      model <- torch::nn_sequential(torch::nn_linear(in_dim, hidden_dim), torch::nn_relu(), torch::nn_linear(hidden_dim, out_dim))
    } else {
      model <- torch::nn_sequential(torch::nn_linear(in_dim, hidden_dim), torch::nn_relu(), torch::nn_batch_norm1d(hidden_dim), torch::nn_linear(hidden_dim, out_dim))
    }
  } else {
    if (!batch_norm) {
      hid <- torch::nn_sequential(torch::nn_linear(hidden_dim, hidden_dim), torch::nn_relu())
      if (num_layer > 3) for (lay in 3:num_layer) hid <- torch::nn_sequential(hid, torch::nn_sequential(torch::nn_linear(hidden_dim, hidden_dim), torch::nn_relu()))
      model <- torch::nn_sequential(torch::nn_sequential(torch::nn_linear(in_dim, hidden_dim), torch::nn_relu()), hid, torch::nn_linear(hidden_dim, out_dim))
    } else {
      hid <- torch::nn_sequential(torch::nn_linear(hidden_dim, hidden_dim), torch::nn_relu(), torch::nn_batch_norm1d(hidden_dim))
      if (num_layer > 3) for (lay in 3:num_layer) hid <- torch::nn_sequential(hid, torch::nn_sequential(torch::nn_linear(hidden_dim, hidden_dim), torch::nn_relu(), torch::nn_batch_norm1d(hidden_dim)))
      model <- torch::nn_sequential(torch::nn_sequential(torch::nn_linear(in_dim, hidden_dim), torch::nn_relu(), torch::nn_batch_norm1d(hidden_dim)), hid, torch::nn_linear(hidden_dim, out_dim))
    }
  }
  model$train()
  optimizer <- torch::optim_adam(model$parameters, lr = lr)

  lossvec <- matrix(nrow = num_epochs, ncol = 3)
  colnames(lossvec) <- c("energy-loss", "Term 1", "Term 2")
  row_indices <- seq_len(n)

  for (epoch in seq_len(num_epochs)) {
    epoch_losses <- c(0, 0, 0)
    perm_rows <- sample(row_indices, replace = FALSE)

    for (b in seq_len(num_batches)) {
      optimizer$zero_grad()

      curr_rows <- perm_rows[((b - 1) * eff_batch_size + 1):min(b * eff_batch_size, n)]
      B_rows <- length(curr_rows)

      curr_idx <- unlist(i_to_terms[curr_rows], use.names = FALSE)
      if (length(curr_idx) == 0) next
      B_terms <- length(curr_idx)

      batch_i <- i_indices_all[curr_idx]
      X_base <- dat_t[batch_i, ]$repeat_interleave(B, dim = 1)            # (B_terms*B, d)
      r_batch <- r_vectors_all_t[curr_idx, ]$repeat_interleave(B, dim = 1) # (B_terms*B, d)
      T_r_mask <- T_r_mask_all_t[curr_idx, ]$repeat_interleave(B, dim = 1) # (B_terms*B, d)

      Y_true_batch <- torch::torch_where(T_r_mask, X_base, torch::torch_zeros_like(X_base))
      yt_r <- Y_true_batch$view(c(B_terms, B, d))

      E <- torch::torch_randn(c(B_terms * B, d))
      X_input <- torch::torch_where(r_batch == 1, X_base, E)
      xt <- torch::torch_cat(list(X_input, r_batch), dim = 2)             # (B_terms*B, 2d)

      yhat_all <- model(xt)
      yhat_r <- (yhat_all * T_r_mask$to(dtype = torch::torch_float()))$view(c(B_terms, B, d))

      # Term 1: mean over B of ||yhat - y||
      d1 <- torch::torch_pow(torch::torch_norm(yt_r - yhat_r, p = 2, dim = 3), beta)
      s1_term <- torch::torch_mean(d1, dim = 2) # (B_terms)

      # Term 2: all-pairs spread within yhat_r, scaled by B/(B-1)
      if (B > 1) {
        pair_dist <- torch::torch_pow(torch::torch_cdist(yhat_r, yhat_r, p = 2), beta) # (B_terms, B, B)
        s2_term <- torch::torch_mean(pair_dist, dim = c(2, 3)) * (B / (B - 1))         # (B_terms)
      } else {
        s2_term <- torch::torch_zeros_like(s1_term)
      }

      Tr_counts <- Tr_counts_all_t[curr_idx]
      loss <- torch::torch_sum((s1_term - 0.5 * s2_term) / Tr_counts) / B_rows

      loss$backward()
      optimizer$step()

      epoch_losses <- epoch_losses + c(
        loss$item(),
        (torch::torch_sum(s1_term / Tr_counts) / B_rows)$item(),
        (torch::torch_sum(s2_term / Tr_counts) / B_rows)$item()
      )
    }

    lossvec[epoch, ] <- epoch_losses / num_batches
    if (!silent) {
      cat(sprintf(
        "\rEpoch [%4d/%4d] | loss: %.4f | T1: %.4f | T2: %.4f",
        epoch, num_epochs, lossvec[epoch, 1], lossvec[epoch, 2], lossvec[epoch, 3]
      ))
      utils::flush.console()
      if (epoch == 1 || epoch %% print_every_nepoch == 0 || epoch == num_epochs) cat("\n")
    }
  }

  if (batch_norm) model$train(mode = FALSE)

  multiemputor <- function(x_new, m = 5) {
    x_new <- as.matrix(x_new)
    n_new <- nrow(x_new)
    d_new <- ncol(x_new)
    out <- array(NA_real_, dim = c(m, n_new, d_new))
    model$eval()

    if (mechanism != "tree") {
      r_mat <- 1 * (!is.na(x_new))
      r_tensor <- torch::torch_tensor(r_mat, dtype = torch::torch_float())

      x_base <- x_new
      x_base[is.na(x_base)] <- 0
      x_base_tensor <- torch::torch_tensor(x_base, dtype = torch::torch_float())

      torch::with_no_grad({
        for (t in seq_len(m)) {
          eps <- torch::torch_randn(c(n_new, d_new))
          eps_masked <- eps * (1 - r_tensor)
          x_batch <- x_base_tensor + eps_masked
          xt <- torch::torch_cat(list(x_batch, r_tensor), dim = 2)
          emp_all <- model(xt)
          X_imp_m <- x_base_tensor + (emp_all * (1 - r_tensor))
          out[t, , ] <- as.matrix(X_imp_m)
        }
      })
      return(out)
    }

    # mechanism = "tree"
    parent_map <- batch_metadata$parent_map
    root <- batch_metadata$root
    max_steps <- d_new * 2

    torch::with_no_grad({
      for (t in seq_len(m)) {
        x_imp <- x_new
        r_curr <- 1 * (!is.na(x_imp))
        x_imp[is.na(x_imp)] <- 0

        for (step in seq_len(max_steps)) {
          curr_pat <- apply(r_curr, 1, vec_to_pattern)
          active <- which(curr_pat != root)
          if (length(active) == 0) break

          child_pat <- curr_pat[active]
          if (any(!(child_pat %in% names(parent_map)))) {
            bad <- unique(child_pat[!(child_pat %in% names(parent_map))])
            stop(sprintf("Tree graph has no parent for pattern(s): %s", paste(bad, collapse = ", ")), call. = FALSE)
          }
          parent_pat <- unname(parent_map[child_pat])
          parent_mat <- do.call(rbind, lapply(parent_pat, pattern_to_vec))
          target_mask_mat <- parent_mat - r_curr[active, , drop = FALSE]

          x_base_tensor <- torch::torch_tensor(x_imp[active, , drop = FALSE], dtype = torch::torch_float())
          r_tensor <- torch::torch_tensor(r_curr[active, , drop = FALSE], dtype = torch::torch_float())
          eps <- torch::torch_randn(c(length(active), d_new))
          x_batch <- torch::torch_where(r_tensor == 1, x_base_tensor, eps)
          xt <- torch::torch_cat(list(x_batch, r_tensor), dim = 2)
          emp_all <- as.matrix(model(xt))

          idx <- which(target_mask_mat == 1, arr.ind = TRUE)
          if (nrow(idx) > 0) {
            x_imp[cbind(active[idx[, 1]], idx[, 2])] <- emp_all[idx]
            r_curr[cbind(active[idx[, 1]], idx[, 2])] <- 1
          }
        }

        if (any(apply(r_curr, 1, vec_to_pattern) != root)) {
          stop("Sequential tree imputation did not reach the complete-case root. Check tree_edges.", call. = FALSE)
        }
        out[t, , ] <- x_imp
      }
    })
    out
  }

  list(model = model, emputor = multiemputor, lossvec = lossvec)
}
