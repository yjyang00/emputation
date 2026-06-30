#' Find response patterns strictly dominated by a given pattern
#'
#' Given a response pattern `Ri` (a binary string, e.g. `"1011"`) and a set of
#' candidate patterns, returns the subset of `set` that are coordinatewise
#' `<= Ri` and strictly smaller in at least one coordinate. Used to build the
#' MCAR masking lookup table in [emputation()].
#'
#' @param Ri A single binary-string response pattern.
#' @param set A character vector of binary-string response patterns to test.
#'
#' @return A character vector: the subset of `set` strictly dominated by `Ri`.
#' @keywords internal
#' @noRd
patterns_less_than <- function(Ri, set) {
  Ri_vec <- as.numeric(strsplit(Ri, "")[[1]])
  Rset_mat <- apply(do.call(rbind, strsplit(set, "")), 2, as.numeric)
  cond1 <- apply(Rset_mat, 1, function(r) all(r <= Ri_vec))
  cond2 <- apply(Rset_mat, 1, function(r) any(r < Ri_vec)) # strictly smaller
  set[cond1 & cond2]
}

#' Build the full MCAR pattern-domination lookup list
#'
#' @param set A character vector of unique binary-string response patterns
#'   observed in the data.
#'
#' @return A named list, one element per pattern in `set`, each containing the
#'   patterns strictly dominated by it.
#' @keywords internal
#' @noRd
all_patterns_less_than <- function(set) {
  out <- lapply(seq_along(set), function(i) patterns_less_than(set[i], set))
  names(out) <- set
  out
}

#' Convert a binary-string response pattern to an integer vector
#'
#' @param x A binary-string pattern (e.g. `"101"`) or an already-numeric/
#'   integer vector, which is returned coerced to integer.
#'
#' @return An integer vector of 0/1 entries.
#' @keywords internal
#' @noRd
pattern_to_vec <- function(x) {
  if (is.numeric(x) || is.integer(x)) return(as.integer(x))
  as.integer(strsplit(as.character(x), "")[[1]])
}

#' Convert a 0/1 vector to a binary-string response pattern
#'
#' @param x A numeric or integer 0/1 vector.
#'
#' @return A single binary-string pattern (e.g. `"101"`).
#' @keywords internal
#' @noRd
vec_to_pattern <- function(x) paste0(as.integer(x), collapse = "")

#' Validate a tree graph of response patterns
#'
#' Checks that `tree_edges` is a well-formed data frame describing a tree
#' (in fact a forest rooted at the complete-case pattern `root`) over binary
#' response patterns of length `p`: correct columns, correct pattern length,
#' binary alphabet, no edges into the root, unique parent per child, each
#' parent strictly dominating its child coordinatewise, no cycles, and every
#' pattern has a path to `root`.
#'
#' @param tree_edges A data frame with character/coercible-to-character
#'   columns `child` and `parent`, each a binary-string response pattern of
#'   length `p`.
#' @param p Integer; the expected length of each pattern.
#' @param root The complete-case pattern (all `1`s) of length `p`.
#'
#' @return `tree_edges` with `child`/`parent` coerced to character, invisibly
#'   validated (the function is called for its error-checking side effect).
#' @keywords internal
#' @noRd
validate_tree_edges <- function(tree_edges, p, root = paste0(rep(1, p), collapse = "")) {
  if (is.null(tree_edges)) {
    stop("For mechanism = \"tree\", please provide tree_edges with columns 'child' and 'parent'.", call. = FALSE)
  }
  if (!is.data.frame(tree_edges) || !all(c("child", "parent") %in% names(tree_edges))) {
    stop("tree_edges must be a data.frame with columns named 'child' and 'parent'.", call. = FALSE)
  }
  tree_edges$child <- as.character(tree_edges$child)
  tree_edges$parent <- as.character(tree_edges$parent)
  all_pat <- c(tree_edges$child, tree_edges$parent)
  if (any(nchar(all_pat) != p)) {
    stop("All tree graph patterns must have length equal to length(tree_vars) (or ncol(dat) if tree_vars = NULL).", call. = FALSE)
  }
  if (any(!grepl("^[01]+$", all_pat))) {
    stop("All tree graph patterns must be binary strings, e.g. '010'.", call. = FALSE)
  }
  if (any(tree_edges$child == root)) {
    stop("The complete-case root pattern should not appear as a child.", call. = FALSE)
  }
  if (any(duplicated(tree_edges$child))) {
    stop("Each incomplete child pattern must have exactly one parent.", call. = FALSE)
  }
  for (j in seq_len(nrow(tree_edges))) {
    child_vec <- pattern_to_vec(tree_edges$child[j])
    parent_vec <- pattern_to_vec(tree_edges$parent[j])
    if (!all(parent_vec >= child_vec) || !any(parent_vec > child_vec)) {
      stop(sprintf(
        "Invalid tree edge %s -> %s: parent must strictly dominate child coordinatewise.",
        tree_edges$child[j], tree_edges$parent[j]
      ), call. = FALSE)
    }
  }
  parent_map <- stats::setNames(tree_edges$parent, tree_edges$child)
  for (child in tree_edges$child) {
    seen <- character(0)
    cur <- child
    while (cur != root) {
      if (cur %in% seen) {
        stop(sprintf("Cycle detected in tree graph starting from pattern %s.", child), call. = FALSE)
      }
      seen <- c(seen, cur)
      nxt <- unname(parent_map[cur])
      if (is.na(nxt)) {
        stop(sprintf("Pattern %s does not have a path to the root %s.", cur, root), call. = FALSE)
      }
      cur <- nxt
    }
  }
  tree_edges
}

#' Expand tree-graph edges defined on a subset of variables to full length
#'
#' `tree_edges` is typically specified only over the variables that actually
#' have missingness (`tree_vars`), e.g. as 3-bit patterns. This expands each
#' pattern to length `d` (the full number of columns in `dat`) by setting all
#' non-tree variables to observed (`1`).
#'
#' @param tree_edges A data frame with columns `child`/`parent`, patterns
#'   defined over `tree_vars` only.
#' @param tree_vars `NULL` (use all `d` columns), a character vector of
#'   column names in `dat`, or an integer vector of column indices.
#' @param dat_colnames Column names of the full data matrix (or `NULL`).
#' @param d Integer; total number of columns in the full data matrix.
#'
#' @return A list with elements `tree_edges_full` (patterns of length `d`),
#'   `tree_edges_small` (the validated input, patterns of length
#'   `length(tree_vars_idx)`), `tree_vars_idx` (integer column indices), and
#'   `tree_vars` (names or indices, for display).
#' @keywords internal
#' @noRd
expand_tree_edges_to_full <- function(tree_edges, tree_vars, dat_colnames, d) {
  if (is.null(tree_vars)) {
    tree_vars_idx <- seq_len(d)
  } else if (is.character(tree_vars)) {
    if (is.null(dat_colnames)) stop("tree_vars is character, but dat has no column names.", call. = FALSE)
    tree_vars_idx <- match(tree_vars, dat_colnames)
    if (any(is.na(tree_vars_idx))) {
      stop(sprintf("tree_vars not found in dat: %s", paste(tree_vars[is.na(tree_vars_idx)], collapse = ", ")), call. = FALSE)
    }
  } else {
    tree_vars_idx <- as.integer(tree_vars)
    if (any(is.na(tree_vars_idx)) || any(tree_vars_idx < 1) || any(tree_vars_idx > d)) {
      stop("tree_vars must be column names or valid column indices.", call. = FALSE)
    }
  }

  p <- length(tree_vars_idx)
  tree_edges_small <- validate_tree_edges(tree_edges, p = p, root = paste0(rep(1, p), collapse = ""))

  expand_one <- function(s) {
    v <- rep(1L, d)
    v[tree_vars_idx] <- pattern_to_vec(s)
    vec_to_pattern(v)
  }

  tree_edges_full <- data.frame(
    child = vapply(tree_edges_small$child, expand_one, character(1)),
    parent = vapply(tree_edges_small$parent, expand_one, character(1)),
    stringsAsFactors = FALSE
  )

  list(
    tree_edges_full = tree_edges_full,
    tree_edges_small = tree_edges_small,
    tree_vars_idx = tree_vars_idx,
    tree_vars = if (is.null(dat_colnames)) tree_vars_idx else dat_colnames[tree_vars_idx]
  )
}

#' Walk a tree graph from a pattern up to the root
#'
#' @param r A binary-string response pattern.
#' @param parent_map A named character vector/list mapping child pattern to
#'   parent pattern.
#' @param root The complete-case root pattern.
#'
#' @return A character vector: the path from `r` to `root` inclusive.
#' @keywords internal
#' @noRd
path_to_root <- function(r, parent_map, root) {
  out <- as.character(r)
  cur <- as.character(r)
  while (cur != root) {
    nxt <- unname(parent_map[cur])
    if (is.na(nxt)) {
      stop(sprintf("Pattern %s has no parent in tree graph.", utils::tail(out, 1)), call. = FALSE)
    }
    cur <- nxt
    out <- c(out, cur)
  }
  out
}
