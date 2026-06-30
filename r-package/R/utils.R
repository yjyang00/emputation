#' Validate top-level arguments to emputation()
#'
#' @inheritParams emputation
#' @param d Number of columns in `dat` (after data.frame coercion).
#' @keywords internal
#' @noRd
check_emputation_args <- function(mechanism, B, hidden_dim, num_layer, num_epochs,
                                   lr, beta, batch_size, n, d, tree_edges) {
  if (!is.character(mechanism) || length(mechanism) != 1 ||
      !mechanism %in% c("mcar", "ccmv", "tree")) {
    stop("mechanism must be one of \"mcar\", \"ccmv\", or \"tree\".", call. = FALSE)
  }
  if (mechanism == "tree" && is.null(tree_edges)) {
    stop("mechanism = \"tree\" requires tree_edges (a data.frame with columns 'child' and 'parent').", call. = FALSE)
  }
  if (!is.numeric(B) || length(B) != 1 || B < 1 || B != round(B)) {
    stop("B must be a positive integer.", call. = FALSE)
  }
  if (B == 1) {
    warning("B = 1 disables the pairwise spread term of the energy score. B should be greater than or equal to 2 (default B = 2).", call. = FALSE)
  }
  if (!is.numeric(hidden_dim) || hidden_dim < 1) stop("hidden_dim must be a positive integer.", call. = FALSE)
  if (!is.numeric(num_layer) || num_layer < 1) stop("num_layer must be a positive integer.", call. = FALSE)
  if (!is.numeric(num_epochs) || num_epochs < 1) stop("num_epochs must be a positive integer.", call. = FALSE)
  if (!is.numeric(lr) || lr <= 0) stop("lr must be a positive number.", call. = FALSE)
  if (!is.numeric(beta) || beta <= 0 || beta >= 2) stop("beta must be in (0, 2).", call. = FALSE)
  if (!is.null(batch_size)) {
    if (!is.numeric(batch_size) || batch_size < 1 || batch_size != round(batch_size)) {
      stop("batch_size must be NULL (full batch) or a positive integer.", call. = FALSE)
    }
    if (batch_size > n) {
      warning("batch_size > nrow(dat); using the full data as a single batch instead.", call. = FALSE)
    }
  }
  if (d < 1) stop("dat must have at least one column.", call. = FALSE)
  invisible(TRUE)
}
