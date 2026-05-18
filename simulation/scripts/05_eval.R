# scripts/05_eval.R
# Reads:
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###.rda                 (data_full, dat_NA, meta)
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###_emputation.rda      (emp_complete)
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###_gain.csv            (GAIN imputed matrix)  [optional]
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###_missForest.rda       (mf_imp)              [optional]
#
# Writes:
#   results/<dataset>/dgp_<dgp>/train_<train>/res_###.rda           

source("scripts/setup.R")
parse_args = function(args) {
  out = list()
  if (length(args) == 0) return(out)
  
  i = 1
  while (i <= length(args)) {
    key = args[i]
    if (!startsWith(key, "--")) stop("Invalid argument: ", key)
    if (i == length(args)) stop("Missing value for: ", key)
    val = args[i + 1]
    key = sub("^--", "", key)
    out[[key]] = val
    i = i + 2
  }
  out
}

args = commandArgs(trailingOnly = TRUE)
opt  = parse_args(args)

if (is.null(opt$scenario_id) || is.null(opt$iter)) {
  stop("Usage: Rscript scripts/05_eval.R --scenario_id <int> --iter <int> [--overwrite 0/1]")
}

scenario_id = as.integer(opt$scenario_id)
iter        = as.integer(opt$iter)
overwrite   = if (!is.null(opt$overwrite)) as.integer(opt$overwrite) else 0

sc = get_scenario(scenario_id)
dataset = tolower(sc$dataset)
dgp     = tolower(sc$dgp)
train   = tolower(sc$train)


# paths
base_file = iter_base_rda(dataset, dgp, train, iter)
emp_file  = iter_emputation_rda(dataset, dgp, train, iter)
gain_file = iter_gain_csv(dataset, dgp, train, iter)
mf_file   = iter_missforest_rda(dataset, dgp, train, iter)

out_dir  = results_scenario_dir(dataset, dgp, train)
out_file = eval_shard_rda(dataset, dgp, train, iter)

ensure_dir(out_dir)

if (!file.exists(base_file)) stop("Missing base artifact: ", base_file)
if (!file.exists(emp_file))  stop("Missing emputation artifact: ", emp_file)

if (file.exists(out_file) && overwrite == 0) {
  cat("Eval shard exists; skipping (overwrite=0):", out_file, "\n")
  quit(save = "no", status = 0)
}

cat("Evaluating one iteration\n")
cat("  scenario:", scenario_tag(dataset, dgp, train), "\n")
cat("  iter:", iter, "\n")

# load
load(base_file)   
load(emp_file)  

true_mat = as.matrix(data_full)
mask = is.na(dat_NA)
dat_NA_df = as.data.frame(dat_NA)

m_imp = cfg$m_imp

# Helper functions
average_imputations = function(imps_list) {
  mats = lapply(imps_list, function(x) as.matrix(x))
  Reduce("+", mats) / length(mats)
}

calc_errors_scaled = function(true_mat, imp_mat, mask) {
  mu = apply(true_mat, 2, mean)
  sigma = apply(true_mat, 2, sd)
  
  t_scaled = scale(true_mat, center = mu, scale = sigma)
  i_scaled = scale(as.matrix(imp_mat), center = mu, scale = sigma)
  
  err = as.numeric(t_scaled[mask] - i_scaled[mask])
  c(RMSE = sqrt(mean(err^2)), MAE = mean(abs(err)))
}

calc_madc = function(true_mat, imps_list) {
  cor_true = cor(true_mat, use = "pairwise.complete.obs")
  cor_list = lapply(imps_list, function(x) cor(as.matrix(x), use = "everything"))
  cor_avg  = Reduce("+", cor_list) / length(cor_list)
  mean(abs(cor_true[lower.tri(cor_true)] - cor_avg[lower.tri(cor_avg)]))
}

calc_energy_distance = function(true_mat, imps_list) {
  mu = apply(true_mat, 2, mean)
  sigma = apply(true_mat, 2, sd)
  t_scaled = scale(true_mat, center = mu, scale = sigma)
  
  ed_vec = sapply(imps_list, function(imp_i) {
    i_scaled = scale(as.matrix(imp_i), center = mu, scale = sigma)
    energy::eqdist.e(rbind(t_scaled, i_scaled), c(nrow(t_scaled), nrow(i_scaled)))
  })
  mean(ed_vec)
}

calc_mmd2_kernlab = function(true_mat, imps_list) {
  t_scaled = scale(true_mat)
  center = attr(t_scaled, "scaled:center")
  sc = attr(t_scaled, "scaled:scale")
  
  mmd2_vec = sapply(imps_list, function(imp_i) {
    i_scaled = scale(as.matrix(imp_i), center = center, scale = sc)
    obj = suppressMessages(kernlab::kmmd(t_scaled, i_scaled, kernel = "rbfdot", kpar = "automatic"))
    obj@mmdstats[2]
  })
  mean(mmd2_vec)
}

is_bad_imp = function(mat) {
  m = as.matrix(mat)
  all(is.na(m)) || any(!is.finite(m))
}

# mean
run_mean_imp = function(dat_NA) {
  X_imp = as.data.frame(dat_NA)
  X_imp[] = lapply(X_imp, function(x) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    x
  })
  list(as.data.frame(X_imp))
}

# mice
run_mice = function(dat_NA, m = 10) {
  imp = mice::mice(dat_NA, m = m, printFlag = FALSE)
  lapply(1:m, function(i) mice::complete(imp, i))
}

# em
run_em_gaussian = function(dat_NA, M, reg = 1e-4) {
  X = as.matrix(dat_NA); N = nrow(X); D = ncol(X)
  fit = em_mvn(X, reg = reg)
  
  imps = vector("list", M)
  for (b in 1:M) {
    X_imp = X
    for (n in 1:N) {
      m_idx = which(is.na(X[n, ]))
      o_idx = which(!is.na(X[n, ]))
      if (length(m_idx) > 0) {
        if (length(o_idx) == 0) {
          X_imp[n, m_idx] = MASS::mvrnorm(
            1,
            mu = fit$mu[m_idx],
            Sigma = fit$Sigma[m_idx, m_idx, drop = FALSE] + diag(reg, length(m_idx))
          )
        } else {
          S_oo = fit$Sigma[o_idx, o_idx, drop = FALSE] + diag(reg, length(o_idx))
          S_mo = fit$Sigma[m_idx, o_idx, drop = FALSE]
          S_om = fit$Sigma[o_idx, m_idx, drop = FALSE]
          S_mm = fit$Sigma[m_idx, m_idx, drop = FALSE]
          
          mu_cond = fit$mu[m_idx] + S_mo %*% solve(S_oo, X[n, o_idx] - fit$mu[o_idx])
          Sigma_cond = S_mm - S_mo %*% solve(S_oo, S_om)
          
          X_imp[n, m_idx] = MASS::mvrnorm(1, mu = as.numeric(mu_cond), Sigma = Sigma_cond)
        }
      }
    }
    if (any(!is.finite(X_imp))) {
      for (j in 1:ncol(X_imp)) X_imp[!is.finite(X_imp[, j]), j] = mean(X[, j], na.rm = TRUE)
    }
    imps[[b]] = as.data.frame(X_imp)
  }
  imps
}

# missForest: load if present, otherwise compute with default params
get_missforest_imps = function(dat_NA_df) {
  if (file.exists(mf_file)) {
    load(mf_file) # expects mf_imp
    return(list(as.data.frame(mf_imp)))
  } else {
    cat("missForest cache not found; computing now (default params)...\n")
    mf_fit = missForest::missForest(dat_NA_df)
    list(as.data.frame(mf_fit$ximp))
  }
}

# GAIN
get_gain_imps = function() {
  if (!file.exists(gain_file)) return(NULL)
  g = read.csv(gain_file, check.names = FALSE)
  g = g[, colnames(dat_NA_df), drop = FALSE]
  if (is_bad_imp(g)) return(list(.bad = TRUE))
  list(as.data.frame(g))
}


methods = list()
methods$Emp = emp_complete
methods$MICE = run_mice(dat_NA_df, m = m_imp)
methods$Mean = run_mean_imp(dat_NA_df)
methods$EM   = run_em_gaussian(dat_NA_df, M = m_imp)
methods$MF   = get_missforest_imps(dat_NA_df)
gain_imps = get_gain_imps()
if (!is.null(gain_imps)) methods$GAIN = gain_imps

# evaluate
iter_results = list()

for (m_name in names(methods)) {
  imps = methods[[m_name]]
  if (is.null(imps) || any(sapply(imps, is_bad_imp))) {
    cat("  WARNING: invalid imputation for", m_name, "— skipping\n")
    next
  }
  
  avg_imp_mat = average_imputations(imps)
  errors = calc_errors_scaled(true_mat, avg_imp_mat, mask)
  
  iter_results[[m_name]] = data.frame(
    iter = iter,
    scenario_id = scenario_id,
    dataset = dataset,
    dgp = dgp,
    train = train,
    method = m_name,
    RMSE = errors["RMSE"],
    MAE  = errors["MAE"],
    MADC = calc_madc(true_mat, imps),
    Energy = calc_energy_distance(true_mat, imps),
    MMD2 = calc_mmd2_kernlab(true_mat, imps)
  )
}

row_res = dplyr::bind_rows(iter_results)

meta_eval = list(
  scenario_id = scenario_id,
  dataset = dataset,
  dgp = dgp,
  train = train,
  iter = iter,
  base_file = base_file,
  emp_file = emp_file,
  gain_file = if (file.exists(gain_file)) gain_file else NA,
  mf_file = if (file.exists(mf_file)) mf_file else NA,
  created_at = as.character(Sys.time())
)

save(row_res, meta_eval, file = out_file)

cat("Saved:)\n")
cat("  file:", out_file, "\n")
