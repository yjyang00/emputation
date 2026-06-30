#' Extract the per-epoch training losses from a fitted emputation model
#'
#' @param object An object of class `"emputation"`.
#'
#' @return A data frame with columns `epoch`, `energy_loss`, `term1`, and
#'   `term2`, one row per training epoch.
#'
#' @examples
#' \dontrun{
#' losses <- emputation_losses(fit)
#' tail(losses)
#' }
#'
#' @export
emputation_losses <- function(object) {
  if (!inherits(object, "emputation")) stop("object must be an emputation fit.", call. = FALSE)
  lv <- object$lossvec
  data.frame(
    epoch = seq_len(nrow(lv)),
    energy_loss = lv[, "energy-loss"],
    term1 = lv[, "Term 1"],
    term2 = lv[, "Term 2"]
  )
}

#' Print an emputation fit
#'
#' @param x An object of class `"emputation"`.
#' @param ... Unused; present for S3 method consistency.
#'
#' @return `x`, invisibly.
#' @export
print.emputation <- function(x, ...) {
  n <- x$batch_metadata$n
  d <- x$batch_metadata$d
  last <- utils::tail(x$lossvec, 1)

  cat("<emputation fit>\n")
  cat(sprintf("  mechanism      : %s\n", x$mechanism))
  cat(sprintf("  data           : n = %d, d = %d\n", n, d))
  cat(sprintf("  neural network : %d hidden layer(s) * %d units (batch_norm = %s)\n",
              x$num_layer, x$hidden_dim, x$batch_norm))
  cat(sprintf("  training       : B = %d, lr = %.2e, beta = %.2f, epochs = %d, batch_size = %s\n",
              x$B, x$lr, x$beta, x$num_epochs, if (is.null(x$batch_size)) "full" else x$batch_size))
  cat(sprintf("  final loss     : energy = %.4f (Term1 = %.4f, Term2 = %.4f)\n",
              last[1], last[2], last[3]))
  if (x$mechanism == "tree") {
    cat(sprintf("  tree edges     : %d\n", nrow(x$tree_edges)))
  }
  invisible(x)
}

#' Summarize an emputation fit
#'
#' @param object An object of class `"emputation"`.
#' @param tail_frac Fraction of final epochs (default last 10%%) used to
#'   report a loss-convergence window.
#' @param ... Unused; present for S3 method consistency.
#'
#' @return An object of class `"summary.emputation"`, printed with a method
#'   that reports the same fields as [print.emputation()] plus the number of
#'   response patterns trained on, the number of training terms, and the
#'   mean/range of the loss over the final `tail_frac` of epochs.
#' @export
summary.emputation <- function(object, tail_frac = 0.1, ...) {
  n <- object$batch_metadata$n
  d <- object$batch_metadata$d
  n_epochs <- nrow(object$lossvec)
  tail_n <- max(1, round(tail_frac * n_epochs))
  tail_loss <- object$lossvec[(n_epochs - tail_n + 1):n_epochs, "energy-loss"]

  n_patterns <- nrow(object$batch_metadata$r_pattern_table)
  n_terms <- object$batch_metadata$N_terms

  out <- list(
    mechanism = object$mechanism,
    n = n, d = d,
    n_patterns = n_patterns,
    n_terms = n_terms,
    hidden_dim = object$hidden_dim,
    num_layer = object$num_layer,
    batch_norm = object$batch_norm,
    B = object$B,
    lr = object$lr,
    beta = object$beta,
    num_epochs = object$num_epochs,
    batch_size = object$batch_size,
    final_loss = utils::tail(object$lossvec, 1),
    tail_frac = tail_frac,
    tail_loss_mean = mean(tail_loss),
    tail_loss_range = range(tail_loss)
  )
  class(out) <- "summary.emputation"
  out
}

#' @export
print.summary.emputation <- function(x, ...) {
  cat("<emputation fit summary>\n")
  cat(sprintf("  mechanism        : %s\n", x$mechanism))
  cat(sprintf("  data             : n = %d, d = %d\n", x$n, x$d))
  cat(sprintf("  response patterns: %d (%d training terms)\n", x$n_patterns, x$n_terms))
  cat(sprintf("  neural network   : %d hidden layer(s) * %d units (batch_norm = %s)\n",
              x$num_layer, x$hidden_dim, x$batch_norm))
  cat(sprintf("  training         : B = %d, lr = %.2e, beta = %.2f, epochs = %d, batch_size = %s\n",
              x$B, x$lr, x$beta, x$num_epochs, if (is.null(x$batch_size)) "full" else x$batch_size))
  cat(sprintf("  final loss       : energy = %.4f (Term1 = %.4f, Term2 = %.4f)\n",
              x$final_loss[1], x$final_loss[2], x$final_loss[3]))
  cat(sprintf("  last %.0f%% of epochs : mean energy loss = %.4f (range %.4f - %.4f)\n",
              100 * x$tail_frac, x$tail_loss_mean, x$tail_loss_range[1], x$tail_loss_range[2]))
  invisible(x)
}

#' Plot training loss diagnostics for an emputation fit
#'
#' @param x An object of class `"emputation"`.
#' @param which One of `"all"` (default; energy loss and both components),
#'   `"energy"`, `"term1"`, or `"term2"`.
#' @param log_y Logical; plot the loss on a log scale (useful when losses
#'   span orders of magnitude early in training).
#' @param ... Additional arguments passed to [graphics::matplot()] /
#'   [graphics::plot()].
#'
#' @return `x`, invisibly. Called for its plotting side effect.
#' @export
plot.emputation <- function(x, which = c("all", "energy", "term1", "term2"), log_y = FALSE, ...) {
  which <- match.arg(which)
  lv <- x$lossvec
  epoch <- seq_len(nrow(lv))
  log_arg <- if (log_y) "y" else ""

  if (which == "all") {
    graphics::matplot(
      epoch, lv, type = "l", lty = 1, col = c("black", "steelblue", "firebrick"),
      log = log_arg, xlab = "epoch", ylab = "loss",
      main = sprintf("emputation training loss (%s)", x$mechanism), ...
    )
    graphics::legend("topright", legend = colnames(lv), col = c("black", "steelblue", "firebrick"), lty = 1, bty = "n")
  } else {
    col_name <- switch(which, energy = "energy-loss", term1 = "Term 1", term2 = "Term 2")
    graphics::plot(
      epoch, lv[, col_name], type = "l", log = log_arg,
      xlab = "epoch", ylab = col_name,
      main = sprintf("emputation training loss (%s)", x$mechanism), ...
    )
  }
  invisible(x)
}
