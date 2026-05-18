# Rscript for running a single bootstrap replicate

args = commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript bootstrap_replicate_MI10.R <seed> <output_dir>")
}

seed = as.integer(args[1])
output_dir = args[2]

cat(sprintf("Bootstrap replicate %d | output dir: %s\n", seed, output_dir))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(tidyverse)
  library(mice)
  library(torch)
})

source("emputation/ImpEng.R")
source("emputation/ImpEngfit.R")
source("emputation/dftomat.R")
source("emputation/helper_pattern.R")
source("emputation/predict.R")
source("emputation/memory.R")

nacc_NA = readRDS("nacc_NA.rds")
M_IMP = 10

logit_bound = function(x, lower, upper, eps = 1e-6) {
  x2 = x
  obs = !is.na(x2)
  x2[obs] = pmin(pmax(x2[obs], lower + eps), upper - eps)
  x2[obs] = log((x2[obs] - lower) / (upper - x2[obs]))
  x2
}

inv_logit_bound = function(z, lower, upper) {
  lower + (upper - lower) * plogis(z)
}

postprocess_imp = function(mat, original_raw_df) {
  df           = as.data.frame(mat)
  colnames(df) = colnames(original_raw_df)
  df$BPSYS   = inv_logit_bound(df$BPSYS,   lower = 70,  upper = 230)
  df$NACCBMI = inv_logit_bound(df$NACCBMI, lower = 10,  upper = 100)
  df$NACCGDS = inv_logit_bound(df$NACCGDS, lower = 0,   upper = 15)
  df$BPSYS   = pmin(pmax(df$BPSYS,   70),  230)
  df$NACCBMI = pmin(pmax(df$NACCBMI, 10),  100)
  df$NACCGDS = pmin(pmax(df$NACCGDS,  0),   15)
  df$NACCGDS = as.integer(round(df$NACCGDS))
  obs_mask     = !is.na(original_raw_df)
  df[obs_mask] = original_raw_df[obs_mask]
  df
}

fit_logistic = function(dat_imp, pathology_group) {
  npadnc_levels = if (pathology_group == "resilience") c(2, 3) else c(0, 1)
  
  dat_model = dat_imp %>%
    mutate(
      APOE_e4 = as.integer(NACCAPOE %in% c(2, 4, 5)),
      APOE_e2 = as.integer(NACCAPOE %in% c(3, 5, 6)),
      outcome = as.integer(CDRGLOB == 0)
    ) %>%
    filter(NPADNC %in% npadnc_levels)
  
  fit = tryCatch(
    glm(outcome ~ AGE + SEX + RACE + EDUC +
          APOE_e4 + APOE_e2 +
          NACCBMI + BPSYS + NACCDIUR + NACCGDS,
        data   = dat_model,
        family = binomial(link = "logit")),
    error = function(e) {
      warning(sprintf("glm failed: %s", e$message))
      NULL
    }
  )
  
  if (is.null(fit)) return(NULL)
  coef(fit)
}

pool_coefs = function(coef_list) {
  valid = Filter(Negate(is.null), coef_list)
  if (length(valid) == 0) return(NULL)
  mat = do.call(rbind, valid)   
  colMeans(mat)                 
}

set.seed(seed)
idx = sample(nrow(nacc_NA), replace = TRUE)
dat_boot = nacc_NA[idx, ]

# complete case analysis
dat_cc_res = dat_boot %>%
  filter(NPADNC %in% c(2, 3)) %>%
  filter(complete.cases(NACCBMI, BPSYS, NACCGDS))

dat_cc_resist = dat_boot %>%
  filter(NPADNC %in% c(0, 1)) %>%
  filter(complete.cases(NACCBMI, BPSYS, NACCGDS))

coef_res_cc = fit_logistic(dat_cc_res, "resilience")
coef_resist_cc = fit_logistic(dat_cc_resist, "resistance")

# MICE
dat_mice_vars = dat_boot %>%
  select(AGE, SEX, RACE, EDUC,
         NACCAPOE, NACCBMI, BPSYS, NACCDIUR, NACCGDS, NPADNC, CDRGLOB)

imp_mice = tryCatch(
  mice(dat_mice_vars, m = M_IMP, maxit = 5, printFlag = FALSE, seed = seed),
  error = function(e) { warning(e$message); NULL }
)

if (!is.null(imp_mice)) {
  mice_coef_res_list = vector("list", M_IMP)
  mice_coef_resist_list = vector("list", M_IMP)
  mice_imp_list = vector("list", M_IMP)
  
  for (m in seq_len(M_IMP)) {
    dm = complete(imp_mice, action = m)
    mice_imp_list[[m]] = dm
    mice_coef_res_list[[m]] = fit_logistic(dm, "resilience")
    mice_coef_resist_list[[m]] = fit_logistic(dm, "resistance")
  }
  coef_res_mice = pool_coefs(mice_coef_res_list)
  coef_resist_mice = pool_coefs(mice_coef_resist_list)
} else {
  mice_imp_list = NULL
  coef_res_mice = NULL
  coef_resist_mice = NULL
}
# emp
dat_emp_boot = dat_boot %>%
  mutate(
    BPSYS   = logit_bound(BPSYS, lower = 70, upper = 230),
    NACCBMI = logit_bound(NACCBMI, lower = 10, upper = 100),
    NACCGDS = logit_bound(NACCGDS, lower = 0, upper = 15)
  ) %>%
  select(AGE, SEX, RACE, EDUC, NACCAPOE,
         NACCBMI, BPSYS, NACCDIUR, NACCGDS, NPADNC, CDRGLOB)

emp_fit = tryCatch(
  emputation(
    dat_emp_boot,
    mechanism = "ccmv", # change to 'mcar' for mechanism=mcar
    num_epochs = 1000, M = 2, hidden_dim = 500, num_layer = 3, batch_norm = TRUE,
    lr = 1e-4, standardize = TRUE, silent = TRUE),
  error = function(e) { warning(e$message); NULL }
)

if (!is.null(emp_fit)) {
  emp_raw_list = predict(emp_fit, dat_emp_boot, m = M_IMP)
  emp_coef_res_list = vector("list", M_IMP)
  emp_coef_resist_list = vector("list", M_IMP)
  emp_imp_list = vector("list", M_IMP)
  
  for (m in seq_len(M_IMP)) {
    dm = postprocess_imp(emp_raw_list[[m]], original_raw_df = dat_boot)
    emp_imp_list[[m]] = dm
    emp_coef_res_list[[m]] = fit_logistic(dm, "resilience")
    emp_coef_resist_list[[m]] = fit_logistic(dm, "resistance")
  }
  coef_res_emp = pool_coefs(emp_coef_res_list)
  coef_resist_emp = pool_coefs(emp_coef_resist_list)
} else {
  emp_imp_list = NULL
  coef_res_emp = NULL
  coef_resist_emp = NULL
}

# save
results = list(
  seed   = seed,
  M_imp  = M_IMP,
  coef_res_cc      = coef_res_cc,
  coef_resist_cc   = coef_resist_cc,
  coef_res_mice    = coef_res_mice,
  coef_resist_mice = coef_resist_mice,
  coef_res_emp     = coef_res_emp,
  coef_resist_emp  = coef_resist_emp,
  dat_mice_imp_list = mice_imp_list,
  dat_emp_imp_list  = emp_imp_list
)

out_file = file.path(output_dir, sprintf("boot_%03d.rds", seed))
saveRDS(results, out_file)
