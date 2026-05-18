library(tidyverse)

# load data
boot_res = readRDS("results/boot_res.rds")

pd = position_dodge(width = 1.0)
ggplot(boot_res, aes(x = OR, y = term, fill = method,    
                          shape = method, color = method)) + 
  geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.6, color = "grey55") +
  geom_errorbarh(
    aes(xmin = CI_lower, xmax = CI_upper),
    position = pd,
    height = 0.20,          
    linewidth = 0.5,
    color = "grey30"        
  ) +
  geom_point(
    position = pd,
    size = 2.6,
    stroke = 0.5            
  ) +
  facet_wrap(~ outcome, scales = "free_x") +
  scale_x_log10(
    breaks = c(0.25, 0.5, 1, 2, 4, 8),
    labels = c("0.25", "0.5", "1", "2", "4", "8")
  ) +
  scale_shape_manual(
    name = "Method",
    values = c("CC" = 21,   "MICE" = 22, "Emputation (CCMV)" = 24, "Emputation (MCAR)" = 25)
    ) +
  scale_fill_manual(
    name = "Method",
    values = c(
      "CC" = "#3A7DC9",
      "MICE" = "#B5529A",
      "Emputation (CCMV)" = "#E8861A",
      "Emputation (MCAR)" = "#3AA174"
    )
  ) +
  scale_color_manual(
    name = "Method",
    values = c(
      "CC" = "#3A7DC9",
      "MICE" = "#B5529A",
      "Emputation (CCMV)" = "#E8861A",
      "Emputation (MCAR)" = "#3AA174"
    )
  ) +
  labs(x = "Odds Ratio",y = NULL) +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major.y = element_line(color = "grey82", linewidth = 0.7),
    panel.grid.major.x = element_line(color = "grey88", linewidth = 0.7),
    panel.grid.minor   = element_blank(),
    axis.text.y        = element_text(size = 11, color = "grey20"),
    axis.text.x        = element_text(size = 10, color = "grey20"),
    axis.title.x       = element_text(size = 12, margin = margin(t = 10)),
    strip.text         = element_text(face = "bold", size = 12),
    strip.background   = element_rect(fill = "grey95", color = NA),
    legend.position    = "bottom",
    legend.title       = element_text(face = "bold"),
    panel.spacing      = unit(1.5, "lines")  # more space between facets
  )
