# Rscript for sensitivity analysis

args = commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript sensitivity.R <seed> <output_dir>")
seed = as.integer(args[1])
output_dir = args[2]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(tidyverse)
  library(torch)
})

source("emputation/ImpEng.R")
source("emputation/ImpEngfit.R")
source("emputation/dftomat.R")
source("emputation/helper_pattern.R")
source("emputation/predict.R")
source("emputation/memory.R")


nacc_NA = readRDS("nacc_NA.rds")
M_SENS = 500
K = 10
rho_grid = seq(-1, 1, length.out = 11)

# Helper functions
logit_bound = function(x, lower, upper, eps = 1e-6) {
  x2 = x; obs = !is.na(x2)
  x2[obs] = pmin(pmax(x2[obs], lower + eps), upper - eps)
  x2[obs] = log((x2[obs] - lower) / (upper - x2[obs]))
  x2
}

inv_logit_bound = function(z, lower, upper) { lower + (upper - lower) * plogis(z) }

postprocess_imp = function(mat, original_raw_df) {
  df = as.data.frame(mat); colnames(df) = colnames(original_raw_df)
  df$BPSYS   = inv_logit_bound(df$BPSYS,   lower = 70,  upper = 230)
  df$NACCBMI = inv_logit_bound(df$NACCBMI, lower = 10,  upper = 100)
  df$NACCGDS = inv_logit_bound(df$NACCGDS, lower = 0,   upper = 15)
  df$NACCGDS = as.integer(round(pmin(pmax(df$NACCGDS, 0), 15)))
  obs_mask = !is.na(original_raw_df)
  df[obs_mask] = original_raw_df[obs_mask]; df
}

fit_logistic = function(dat_imp, pathology_group) {
  npadnc_levels = if (pathology_group == "resilience") c(2, 3) else c(0, 1)
  dat_model = dat_imp %>%
    mutate(APOE_e4 = as.integer(NACCAPOE %in% c(2, 4, 5)),
           APOE_e2 = as.integer(NACCAPOE %in% c(3, 5, 6)),
           outcome = as.integer(CDRGLOB == 0)) %>%
    filter(NPADNC %in% npadnc_levels)
  fit = tryCatch(glm(outcome ~ AGE + SEX + RACE + EDUC + APOE_e4 + APOE_e2 +
                       NACCBMI + BPSYS + NACCDIUR + NACCGDS,
                     data = dat_model, family = binomial(link = "logit")),
                 error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  coef(fit)
}

# bootstrap
set.seed(seed)
idx      = sample(nrow(nacc_NA), replace = TRUE)
dat_boot = nacc_NA[idx, ]

dat_emp_boot = dat_boot %>%
  mutate(BPSYS   = logit_bound(BPSYS,   70, 230),
         NACCBMI = logit_bound(NACCBMI, 10, 100),
         NACCGDS = logit_bound(NACCGDS, 0,  15)) %>%
  select(AGE, SEX, RACE, EDUC, NACCAPOE, NACCBMI, BPSYS, NACCDIUR, NACCGDS, NPADNC, CDRGLOB)

emp_fit = emputation(dat_emp_boot, mechanism = "ccmv",
                     M = 2, num_epochs = 1000, hidden_dim = 500, num_layer = 3,
                     batch_norm = TRUE, lr = 1e-4, standardize = TRUE, silent = TRUE)

saveRDS(list(seed = seed, emp_fit = emp_fit, dat_boot = dat_boot, dat_emp_boot = dat_emp_boot),
        file.path(output_dir, sprintf("emp_fit_%03d.rds", seed)))

Yhat_logit = predict(emp_fit, dat_emp_boot, m = M_SENS)
Yhat_std = lapply(Yhat_logit, function(mat){
  sweep(sweep(mat, 2, emp_fit$muX, "-"), 2, emp_fit$sddX, "/")
})
is_miss = is.na(as.matrix(dat_emp_boot))
has_miss = rowSums(is_miss) > 0
mu_ztilde = Reduce("+", Yhat_std) / M_SENS
n = nrow(as.matrix(dat_emp_boot))
dist_mat = matrix(0, nrow = n, ncol = M_SENS)
for (m in 1:M_SENS) {
  dev = Yhat_std[[m]] - mu_ztilde
  dev_sq  = (dev^2) * is_miss
  dist_mat[, m] = rowSums(dev_sq)
}
dist_mat_inc = dist_mat[has_miss, ]
dm_list = vector("list", M_SENS)
for (m in 1:M_SENS) {
  dm_list[[m]] = postprocess_imp(Yhat_logit[[m]], original_raw_df = dat_boot)
}

sens_results = lapply(rho_grid, function(rho) {
  log_w_mat = -rho * dist_mat_inc
  log_w_max = apply(log_w_mat, 1, max)
  log_w_adj = log_w_mat - log_w_max
  w_mat = exp(log_w_adj)
  w_mat = w_mat / rowSums(w_mat)
  
  ess = mean(1 / rowSums(w_mat^2))
  cat("rho =", rho, "| avg ESS:", round(ess, 1), "\n")
  coef_res_k = vector("list", K)
  coef_resist_k = vector("list", K)
  inc_idx = which(has_miss)
  
  for (k in 1:K) {
    dm_k = dm_list[[1]]
    for (ii in seq_along(inc_idx)) {
      i = inc_idx[ii]
      m_draw = sample(M_SENS, size = 1, prob = w_mat[ii, ])
      dm_k[i, ] = dm_list[[m_draw]][i, ]
    }
    coef_res_k[[k]] = fit_logistic(dm_k, "resilience")
    coef_resist_k[[k]] = fit_logistic(dm_k, "resistance")
  }
  
  pool = function(coef_list) {
    valid = !sapply(coef_list, is.null)
    if (!any(valid)) return(NULL)
    colMeans(do.call(rbind, coef_list[valid]))
  }
  
  list(
    rho = rho,
    coef_res = pool(coef_res_k),
    coef_resist = pool(coef_resist_k)
  )
})

# save
saveRDS(list(seed = seed, sens_results = sens_results),
        file.path(output_dir, sprintf("sens_boot_%03d.rds", seed)))
