#' Fit an emputation model
#'
#' Fits a neural imputation model by minimizing the emputation risk
#' (an energy-score-based loss), under a chosen missing-data identification
#' assumption. The fitted object can be passed to [predict.emputation()] to
#' draw multiple imputations, and to the
#' [print()], [summary()], and [plot()] methods for training diagnostics.
#'
#' @param dat A matrix or a data frame with missing values coded as `NA`.
#' @param mechanism One of `"mcar"`, `"ccmv"`, or `"tree"`, selecting the
#'   missing data identification assumption that determines how the
#'   emputation risk masks and selects observations during training:
#'   * `"mcar"`: missing completely at random. 
#'   * `"ccmv"`: complete-case missing variable assumption. 
#'   * `"tree"`: tree graph.
#' @param B Integer; the number of Monte Carlo draws used to approximate the
#'   emputation risk (default is `B = 2`).
#' @param hidden_dim Integer; width of each hidden layer.
#' @param num_layer Integer; number of hidden layers.
#' @param batch_norm Logical; whether to use batch normalization after each
#'   hidden layer.
#' @param num_epochs Integer; number of training epochs.
#' @param lr Numeric; learning rate for the Adam optimizer.
#' @param beta Numeric in `(0, 2)`; the energy score exponent.
#' @param standardize Logical; whether to standardize each column of `dat`. 
#' @param silent Logical; if `FALSE` (default), prints a one-line training
#'   progress update each epoch.
#' @param batch_size `NULL` (default) to train on the full data each epoch,
#'   or a positive integer giving the number of *rows* of `dat` per
#'   mini-batch.
#' @param print_every_nepoch Integer; how often (in epochs) to print a
#'   newline-terminated progress line when `silent = FALSE`.
#' @param tree_edges Required when `mechanism = "tree"`. A data frame with
#'   columns `child` and `parent`, each a binary-string response pattern
#'   (e.g. `"010"`, with `1` = observed) defined over `tree_vars`. 
#' @param tree_vars `NULL` (default; tree patterns are defined over all
#'   columns of `dat`), or a character vector of column names / integer
#'   vector of column indices over which `tree_edges` patterns are defined.
#'   Non-listed columns are treated as always observed in the tree patterns.
#'
#' @return An object of class `"emputation"`, a list with components
#'   including:
#'   \describe{
#'     \item{emputor}{The fitted sampler function used internally by
#'       [predict.emputation()].}
#'     \item{model}{The underlying `torch` `nn_module`.}
#'     \item{lossvec}{A `num_epochs` * 3 matrix of per-epoch training losses
#'       (energy loss and its two component terms); see
#'       [emputation_losses()].}
#'     \item{mechanism, B, hidden_dim, num_layer, batch_norm, num_epochs, lr,
#'       standardize, batch_size}{Echoed fitting arguments.}
#'     \item{muX, sddX}{Observed-data column means/sds used for
#'       standardization.}
#'     \item{Rmat, batch_metadata}{Internal bookkeeping of response patterns
#'       (not normally needed by end users).}
#'   }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 1000
#' X <- matrix(rnorm(n * 5), n, 5)
#' X[sample(n * 5, floor(0.15 * n * 5))] <- NA
#' fit <- emputation(X, mechanism = "mcar", num_epochs = 500, silent = FALSE)
#' print(fit)
#' plot(fit)
#'
#' imputed <- predict(fit, X, m = 5) # list of 5 completed datasets
#' }
#'
#' @export
emputation <- function(dat, mechanism, B = 2, hidden_dim = 500, num_layer = 3,
                        batch_norm = TRUE, num_epochs = 1000, lr = 10^(-4), beta = 1,
                        standardize = TRUE, silent = FALSE,
                        batch_size = NULL, print_every_nepoch = 100,
                        tree_edges = NULL, tree_vars = NULL) {

  if (is.data.frame(dat)) {
    if (any(sapply(dat, is.factor))) {
      warning("dat contains factor variables. Mapping to numeric values. Dummy variables would need to be created explicitly by the user.", call. = FALSE)
    }
    dat <- dftomat(dat)
  }

  if (is.vector(dat) && !is.numeric(dat)) dat <- as.numeric(dat)
  if (is.vector(dat) && is.numeric(dat)) dat <- matrix(dat, ncol = 1)
  if (!is.matrix(dat) || !is.numeric(dat)) {
    stop("dat must be a numeric matrix, numeric vector, or data.frame coercible to one.", call. = FALSE)
  }

  check_emputation_args(
    mechanism = mechanism, B = B, hidden_dim = hidden_dim, num_layer = num_layer,
    num_epochs = num_epochs, lr = lr, beta = beta, batch_size = batch_size,
    n = nrow(dat), d = ncol(dat), tree_edges = tree_edges
  )

  if (all(stats::complete.cases(dat))) {
    warning("dat has no missing values; emputation will still train, but there is nothing to impute.", call. = FALSE)
  }

  muX <- apply(dat, 2, function(col) mean(col, na.rm = TRUE))
  sddX <- apply(dat, 2, function(col) stats::sd(col, na.rm = TRUE))
  if (any(sddX <= 0)) {
    warning("Some variables have zero variance on observed data -- results might be unreliable.", call. = FALSE)
    sddX <- pmax(sddX, 10^(03))
  }

  dat_std <- dat
  if (standardize) {
    dat_std <- sweep(sweep(dat, 2, muX, FUN = "-"), 2, sddX, FUN = "/")
  }

  d <- ncol(dat)
  Rmat <- 1 * (!is.na(dat))
  R <- apply(Rmat, 1, vec_to_pattern)

  if (mechanism == "mcar") {
    Rset <- unique(R)
    Rls <- all_patterns_less_than(Rset) # look-up list
    R_type_index <- match(R, names(Rls))

    unique_r_strs <- unique(unlist(Rls)) # lookup table
    r_pattern_table <- do.call(rbind, lapply(unique_r_strs, pattern_to_vec))
    r_pattern_index <- stats::setNames(seq_along(unique_r_strs), unique_r_strs)

    N_terms_max <- sum(sapply(R_type_index, function(idx) length(Rls[[idx]])))
    i_indices <- integer(N_terms_max)
    r_idx_vec <- integer(N_terms_max)
    Tr_counts <- integer(N_terms_max)
    k <- 1
    for (i in seq_len(nrow(dat))) {
      Ri_vec <- as.integer(Rmat[i, ])
      rls <- Rls[[R_type_index[i]]]
      if (length(rls) == 0) next

      for (r_str in rls) {
        r_vec <- r_pattern_table[r_pattern_index[[r_str]], ]
        i_indices[k] <- i
        r_idx_vec[k] <- r_pattern_index[[r_str]]
        Tr_counts[k] <- sum(Ri_vec - r_vec)
        k <- k + 1
      }
    }
  
    batch_metadata <- list(
      N_terms = k - 1,
      i_indices = i_indices[seq_len(k - 1)],
      r_pattern_table = r_pattern_table,
      r_idx_vec = r_idx_vec[seq_len(k - 1)],
      Tr_counts = Tr_counts[seq_len(k - 1)],
      d = d,
      n = nrow(dat)
    )
  } else if (mechanism == "ccmv") {
    Rset <- unique(R)
    Rcc <- paste0(rep(1, d), collapse = "")
    Rset <- Rset[Rset != Rcc]
    if (length(Rset) == 0) stop("mechanism = \"ccmv\" found no incomplete response patterns in dat.", call. = FALSE)
    r_pattern_table <- do.call(rbind, lapply(Rset, pattern_to_vec))
    r_pattern_index <- stats::setNames(seq_along(Rset), Rset)

    N_cc <- sum(rowSums(Rmat) == d)
    if (N_cc == 0) stop("mechanism = \"ccmv\" requires at least one complete-case row in dat.", call. = FALSE)
    N_terms_max <- N_cc * length(Rset)
    i_indices <- integer(N_terms_max)
    r_idx_vec <- integer(N_terms_max)
    Tr_counts <- integer(N_terms_max)
    k <- 1

    for (i in seq_len(nrow(dat))) {
      if (R[i] != Rcc) next

      for (r_str in Rset) {
        r_vec <- r_pattern_table[r_pattern_index[[r_str]], ]
        i_indices[k] <- i
        r_idx_vec[k] <- r_pattern_index[[r_str]]
        Tr_counts[k] <- sum(1 - r_vec)
        k <- k + 1
      }
    }

    batch_metadata <- list(
      N_terms = k - 1,
      i_indices = i_indices[seq_len(k - 1)],
      r_pattern_table = r_pattern_table,
      r_idx_vec = r_idx_vec[seq_len(k - 1)],
      Tr_counts = Tr_counts[seq_len(k - 1)],
      d = d,
      n = nrow(dat)
    )
  } else { # tree
    Rcc <- paste0(rep(1, d), collapse = "")

    tree_info <- expand_tree_edges_to_full(
      tree_edges = tree_edges,
      tree_vars = tree_vars,
      dat_colnames = colnames(dat),
      d = d
    )
    tree_edges_full <- validate_tree_edges(tree_info$tree_edges_full, p = d, root = Rcc)
    parent_map <- stats::setNames(tree_edges_full$parent, tree_edges_full$child)

    Rset <- unique(R)
    Rset <- Rset[Rset != Rcc] 
    Rset <- Rset[Rset %in% tree_edges_full$child]
    if (length(Rset) == 0) stop("No incomplete response patterns in dat are covered by tree_edges/tree_vars.", call. = FALSE)

    uncovered <- setdiff(unique(R[R != Rcc]), tree_edges_full$child)
    if (length(uncovered) > 0) {
      warning(sprintf(
        "Some incomplete response patterns in dat are not covered by the tree and will not be trained on or imputed: %s",
        paste(uncovered, collapse = ", ")
      ), call. = FALSE)
    }

    r_pattern_table <- do.call(rbind, lapply(Rset, pattern_to_vec))
    r_pattern_index <- stats::setNames(seq_along(Rset), Rset)
    parent_for_r <- parent_map[Rset]
    parent_pattern_table <- do.call(rbind, lapply(parent_for_r, pattern_to_vec))
    T_mask_table <- parent_pattern_table - r_pattern_table # PA(r) - r
    Tr_counts_by_r <- rowSums(T_mask_table)

    if (any(Tr_counts_by_r <= 0)) stop("Invalid tree graph: every tree edge must add at least one observed coordinate.", call. = FALSE)
    missing_parent <- setdiff(unique(parent_for_r), unique(R))
    if (length(missing_parent) > 0) {
      warning(sprintf(
        "Some parent patterns have no rows in dat and will contribute no training terms: %s",
        paste(missing_parent, collapse = ", ")
      ), call. = FALSE)
    }

    N_terms_max <- sum(sapply(R, function(pat) sum(parent_for_r == pat)))
    i_indices <- integer(N_terms_max)
    r_idx_vec <- integer(N_terms_max)
    Tr_counts <- integer(N_terms_max)
    k <- 1

    for (i in seq_len(nrow(dat))) {
      Ri_str <- R[i]
      child_idx <- which(parent_for_r == Ri_str)
      if (length(child_idx) == 0) next
      for (j in child_idx) {
        i_indices[k] <- i
        r_idx_vec[k] <- j
        Tr_counts[k] <- Tr_counts_by_r[j]
        k <- k + 1
      }
    }

    batch_metadata <- list(
      N_terms = k - 1,
      i_indices = i_indices[seq_len(k - 1)],
      r_pattern_table = r_pattern_table,
      r_idx_vec = r_idx_vec[seq_len(k - 1)],
      Tr_counts = Tr_counts[seq_len(k - 1)],
      T_mask_table = T_mask_table,
      parent_pattern_table = parent_pattern_table,
      Rset = Rset,
      tree_edges = tree_edges_full,
      tree_edges_small = tree_info$tree_edges_small,
      tree_vars = tree_info$tree_vars,
      tree_vars_idx = tree_info$tree_vars_idx,
      parent_map = parent_map,
      root = Rcc,
      d = d,
      n = nrow(dat)
    )
    if (batch_metadata$N_terms == 0) stop("Tree graph created zero training terms. Check whether parent patterns exist in dat.", call. = FALSE)
  }

  message("Metadata preparation complete. Starting emputation...")
  emp <- emputationfit(
    dat = dat_std, mechanism = mechanism, batch_metadata = batch_metadata,
    Rmat = Rmat, B = B, hidden_dim = hidden_dim, num_layer = num_layer,
    batch_norm = batch_norm, num_epochs = num_epochs, lr = lr, beta = beta,
    batch_size = batch_size, silent = silent, print_every_nepoch = print_every_nepoch
  )

  emputor <- list(
    emputor = emp$emputor, lossvec = emp$lossvec, mechanism = mechanism,
    B = B, Rmat = Rmat, batch_metadata = batch_metadata, muX = muX, sddX = sddX,
    standardize = standardize, hidden_dim = hidden_dim, num_layer = num_layer,
    batch_norm = batch_norm, num_epochs = num_epochs, lr = lr, beta = beta,
    model = emp$model, batch_size = batch_size
  )
  if (mechanism == "tree") {
    emputor$tree_edges <- batch_metadata$tree_edges
    emputor$tree_edges_small <- batch_metadata$tree_edges_small
    emputor$tree_vars <- batch_metadata$tree_vars
    emputor$tree_vars_idx <- batch_metadata$tree_vars_idx
  }
  class(emputor) <- "emputation"
  emputor
}
