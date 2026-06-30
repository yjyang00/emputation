#' Convert a data frame to a numeric matrix
#'
#' Converts a data frame into a numeric matrix. Factor and character columns
#' are coerced to numeric (factors via their underlying levels-as-character,
#' characters via `as.factor()` first). This is a thin convenience used
#' internally by [emputation()] and [predict.emputation()]; users who need
#' control over dummy coding of categorical variables should construct their
#' numeric matrix explicitly before calling [emputation()].
#'
#' @param X A data frame to be converted to a numeric matrix.
#'
#' @return A numeric matrix with the same dimensions as `X`.
#'
#' @keywords internal
#' @noRd
dftomat <- function(X) {
  X <- data.frame(lapply(X, function(x) {
    if (is.factor(x)) {
      as.numeric(as.character(x))
    } else if (is.character(x)) {
      as.numeric(as.factor(x))
    } else {
      as.numeric(x)
    }
  }))
  X <- as.matrix(X)
  X
}
