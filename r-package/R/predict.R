#' Draw multiple imputations from a fitted emputation model
#'
#' @param object An object of class `"emputation"`, as returned by
#'   [emputation()].
#' @param Xtest A matrix or data frame, with missing values coded as `NA`. 
#' @param m Integer; number of imputed datasets to draw.
#' @param ... Unused; present for S3 method consistency.
#' @return A list of length `m`, each element an `nrow(Xtest)` *
#'   `ncol(Xtest)` matrix with all `NA`s filled in.
#'
#' @examples
#' \dontrun{
#' imputed <- predict(fit, Xtest, m = 10)
#' }
#'
#' @export
predict.emputation <- function(object, Xtest, m = 10, ...) {
  
  if (is.data.frame(Xtest)) Xtest <- dftomat(Xtest)
  if (is.vector(Xtest) && is.numeric(Xtest)) Xtest <- matrix(Xtest, ncol = 1)
  if (!is.matrix(Xtest) || !is.numeric(Xtest)) {
    stop("Xtest must be a numeric matrix, numeric vector, or data.frame coercible to one.", call. = FALSE)
  }
  if (ncol(Xtest) != length(object$muX)) {
    stop(sprintf("Xtest has %d columns but the model was fit on %d columns.", ncol(Xtest), length(object$muX)), call. = FALSE)
  }
  if (!is.numeric(m) || length(m) != 1 || m < 1 || m != round(m)) {
    stop("m must be a single positive integer.", call. = FALSE)
  }
  
  Xtest_orig <- Xtest
  obs <- !is.na(Xtest_orig)
  
  if (object$standardize) {
    Xtest <- sweep(sweep(Xtest, 2, object$muX, FUN = "-"), 2, object$sddX, FUN = "/")
  }
  
  Yhat1 <- object$emputor(Xtest, m = m)
  emp_list <- lapply(seq_len(m), function(t) Yhat1[t, , ])
  
  if (object$standardize) {
    emp_list <- lapply(emp_list, function(mat) {
      sweep(sweep(mat, 2, object$sddX, "*"), 2, object$muX, "+")
    })
  }
  
  emp_list <- lapply(emp_list, function(mat) {
    mat[obs] <- Xtest_orig[obs]
    mat
  })
  
  emp_list
}
