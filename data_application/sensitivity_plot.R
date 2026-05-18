# =============================================================================
# aggregate_sensitivity.R
# Aggregate sensitivity results across bootstrap replicates and plot
# =============================================================================
library(tidyverse)
library(patchwork)

args       = commandArgs(trailingOnly = TRUE)
output_dir = if (length(args) >= 1) args[1] else "results_sensitivity/bootstrap"

cat("Loading results from:", output_dir, "\n")

# ── Load all bootstrap results ────────────────────────────────────────────────
files = list.files(output_dir, pattern = "sens_boot_.*\\.rds", full.names = TRUE)
cat("Found", length(files), "bootstrap replicates\n")
all_results = lapply(files, readRDS)

# ── Extract coefficients across replicates ────────────────────────────────────
rho_grid = sapply(all_results[[1]]$sens_results, `[[`, "rho")

collect_coefs = function(model) {
  lapply(seq_along(rho_grid), function(rho_idx) {
    do.call(rbind, lapply(all_results, function(boot) {
      coef_vec = boot$sens_results[[rho_idx]][[model]]
      if (is.null(coef_vec)) return(NULL)
      coef_vec
    }))
  })
}

coefs_res    = collect_coefs("coef_res")
coefs_resist = collect_coefs("coef_resist")

# ── Compute OR and 95% bootstrap CI for each rho ─────────────────────────────
summarize_coefs = function(coefs_list, rho_grid) {
  do.call(rbind, lapply(seq_along(rho_grid), function(rho_idx) {
    mat = coefs_list[[rho_idx]]
    if (is.null(mat) || nrow(mat) == 0) return(NULL)
    or    = exp(colMeans(mat))
    ci_lo = exp(apply(mat, 2, quantile, 0.025))
    ci_hi = exp(apply(mat, 2, quantile, 0.975))
    data.frame(rho = rho_grid[rho_idx], term = names(or),
               OR = or, CI_lo = ci_lo, CI_hi = ci_hi)
  }))
}

summary_res    = summarize_coefs(coefs_res,    rho_grid)
summary_resist = summarize_coefs(coefs_resist, rho_grid)

# ── Variables ─────────────────────────────────────────────────────────────────
vars_of_interest = c("AGE","SEX","RACE","EDUC","APOE_e4",
                     "APOE_e2","NACCBMI","BPSYS","NACCDIUR","NACCGDS")

# ── Helper: build one panel (no clipping, free y per facet) ──────────────────
make_panel = function(summary_df, title, color_line, color_fill, line_type = "solid") {
  summary_df %>%
    filter(term %in% vars_of_interest) %>%
    mutate(term = factor(term, levels = vars_of_interest)) %>%
    ggplot(aes(x = rho, y = OR, ymin = CI_lo, ymax = CI_hi)) +
    geom_hline(yintercept = 1, linetype = "dashed",
               colour = "grey70", linewidth = 0.35) +
    geom_ribbon(fill = color_fill, alpha = 0.20, colour = NA) +
    geom_line(colour = color_line, linewidth = 0.7, linetype = line_type) +
    facet_wrap(~ term, scales = "free_y", ncol = 5) +
    scale_x_continuous(breaks = c(-1, 0, 1)) +
    labs(title = title, x = expression(rho), y = "Odds Ratio") +
    theme_bw(base_size = 9) +
    theme(
      strip.text        = element_text(size = 8, face = "bold"),
      strip.background  = element_rect(fill = "grey90", colour = "grey60"),
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = "grey88"),
      plot.title        = element_text(size = 10, face = "bold",
                                       margin = margin(b = 4)),
      axis.title.x      = element_text(size = 8),
      axis.title.y      = element_text(size = 8),
      plot.margin       = margin(4, 8, 4, 4)
    )
}

p_res    = make_panel(summary_res,    "(a) Resilience",
                      color_line = "black",   color_fill = "grey50", line_type = "solid")
p_resist = make_panel(summary_resist, "(b) Resistance",
                      color_line = "grey35",  color_fill = "grey65", line_type = "solid")

p_combined = p_res + p_resist +
  plot_layout(ncol = 2) +
  plot_annotation(
    title   = " ",
    theme   = theme(
      plot.title   = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.caption = element_text(size = 7,  colour = "grey50", hjust = 0)
    )
  )

ggsave(file.path(output_dir, "sensitivity_combined.pdf"),
       p_combined, width = 20, height = 7)
