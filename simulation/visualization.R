library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
library(grid)
library(ggpattern)
library(tools)
library(purrr)

# load data: 20% sim result
results_dir ="simulation/results/results_20" # change to results_40 for 40% sim result
rda_files = list.files(results_dir, pattern = "\\.rda$", full.names = TRUE)

all_summaries = map_dfr(rda_files, function(path) {
  fname     = file_path_sans_ext(basename(path))
  data_name = str_extract(fname, "^[^_]+")
  dgp_name  = str_remove(fname, paste0("^", data_name, "_dgp"))
  
  env = new.env(parent = emptyenv())
  load(path, envir = env)
  
  get("summary_df", envir = env) %>%
    mutate(data = data_name, dgp = dgp_name,
           train = case_when(
             method == "Emp_mcar" ~ "mcar", method == "Emp_ccmv" ~ "ccmv", TRUE ~ NA_character_),
           method = if_else(str_starts(method, "Emp_"), "Emp", method))
})

method_levels = c(
  "EM", "MICE", "MF", "GAIN", "Mean", "Emp_match", "Emp_mis"
)

legend_labels = c(
  "EM"        = "EM",
  "MICE"      = "MICE",
  "MF"        = "MF",
  "GAIN"      = "GAIN",
  "Mean"      = "Mean",
  "Emp_match" = "Emp (reference)",
  "Emp_mis"   = "Emp (mismatched)"
)

method_fills = c(
  "EM"        = "#111111",
  "MICE"      = "#444444",
  "MF"        = "#777777",
  "GAIN"      = "#AAAAAA",
  "Mean"      = "#CCCCCC",
  "Emp_match" = "#FFFFFF",   
  "Emp_mis"   = "#FFFFFF"    
)

method_patterns = c(
  "EM"        = "none",
  "MICE"      = "none",
  "MF"        = "none",
  "GAIN"      = "none",
  "Mean"      = "none",
  "Emp_match" = "stripe",   
  "Emp_mis"   = "none"      
)

method_pattern_angle = c(
  "EM"        = 0,
  "MICE"      = 0,
  "MF"        = 0,
  "GAIN"      = 0,
  "Mean"      = 0,
  "Emp_match" = 45,   
  "Emp_mis"   = 0
)


emp_ref = all_summaries %>%
  filter(method == "Emp") %>%
  mutate(is_ref = case_when(
    dgp == "ccmv" & train == "ccmv" ~ TRUE,
    dgp == "mcar" & train == "mcar" ~ TRUE,
    dgp == "mar"  & train == "mcar" ~ TRUE,
    TRUE ~ FALSE
  )) %>%
  filter(is_ref) %>%
  group_by(data, dgp) %>%
  summarise(
    MADC_ref   = mean(MADC_mean,   na.rm = TRUE),
    Energy_ref = mean(Energy_mean, na.rm = TRUE),
    MMD2_ref   = mean(MMD2_mean,   na.rm = TRUE),
    .groups = "drop"
  )


non_emp = all_summaries %>%
  filter(method != "Emp") %>%
  group_by(data, dgp, method) %>%
  summarise(
    across(c(MADC_mean, Energy_mean, MMD2_mean), mean, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    bar_id = method,
    train = NA_character_
  )

emp_bars = all_summaries %>%
  filter(method == "Emp") %>%
  select(data, dgp, train, MADC_mean, Energy_mean, MMD2_mean) %>%
  mutate(bar_id = case_when(
    dgp == "ccmv" & train == "ccmv" ~ "Emp_match",
    dgp == "mcar" & train == "mcar" ~ "Emp_match",
    dgp == "mar"  & train == "mcar" ~ "Emp_match",
    TRUE ~ "Emp_mis"
  ))

plot_data = bind_rows(non_emp, emp_bars) %>%
  left_join(emp_ref, by = c("data", "dgp")) %>%
  mutate(
    rel_MADC   = MADC_mean   / MADC_ref,
    rel_Energy = Energy_mean / Energy_ref,
    rel_MMD2   = MMD2_mean   / MMD2_ref,
    bar_id     = factor(bar_id, levels = method_levels)
  )

plot_data_filtered = plot_data %>%
  filter(dgp != "mar")

bar_positions = c(
  "EM"        = 1.0,
  "MICE"      = 2.0,
  "MF"        = 3.0,
  "GAIN"      = 4.0,
  "Mean"      = 5.0,
  "Emp_match" = 6.2,
  "Emp_mis"   = 7.2
)

dgp_levels  = c("ccmv", "mcar")
dgp_offsets = c("ccmv" = 0, "mcar" = 9)

plot_long = plot_data_filtered %>%
  select(data, dgp, bar_id, rel_MADC, rel_Energy, rel_MMD2) %>%
  pivot_longer(
    cols = starts_with("rel_"),
    names_to = "metric",
    values_to = "rel_value"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("rel_MADC", "rel_Energy", "rel_MMD2"),
      labels = c("Rel. MADC", "Rel. Energy", "Rel. MMD2")
    ),
    data = factor(data, levels = c("ccpp", "concrete", "wine")),
    dgp  = factor(dgp, levels = dgp_levels),
    bar_pos = bar_positions[as.character(bar_id)] + dgp_offsets[as.character(dgp)]
  )

dgp_tick_pos = dgp_offsets + 4.1
dgp_tick_labels = c(
  "ccmv" = "DGP: ccmv",
  "mcar" = "DGP: mcar"
)

# plot
final_plot = ggplot(
  plot_long,
  aes(
    x = bar_pos,
    y = rel_value,
    fill = bar_id,
    pattern = bar_id,
    pattern_angle = bar_id
  )
) +
  ggpattern::geom_col_pattern(
    width                    = 0.72,
    color                    = "grey15",
    linewidth                = 0.25,
    pattern_fill             = "grey15",
    pattern_colour           = "grey15",
    pattern_density          = 0.05,   
    pattern_spacing          = 0.02,   
    pattern_key_scale_factor = 0.5
  ) + 
  geom_vline(
    xintercept = 8.6,
    linetype = "dotted",
    color = "grey55",
    linewidth = 0.35
  ) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    color = "#CC0000",
    linewidth = 0.28
  ) +
  facet_grid(
    metric ~ data,
    scales = "free_y",
    switch = "y"
  ) +
  scale_fill_manual(
    values = method_fills,
    breaks = method_levels,
    labels = legend_labels
  ) +
  scale_pattern_manual(
    values = method_patterns,
    breaks = method_levels,
    labels = legend_labels
  ) +
  scale_pattern_angle_manual(
    values = method_pattern_angle,
    breaks = method_levels,
    labels = legend_labels
  ) +
  scale_x_continuous(
    breaks = unname(dgp_tick_pos),
    labels = unname(dgp_tick_labels),
    expand = expansion(add = 0.45)
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    trans = scales::pseudo_log_trans(base = 10),
    breaks = function(x) {
      b = c(0, 1, 10, 100, 500, 1000, 2000)
      b[b >= x[1] & b <= x[2]]
    },
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(x = NULL, y = NULL, fill = NULL, pattern = NULL) +
  theme_bw(base_size = 9) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.45, "cm"),
    legend.spacing.x = unit(0.15, "cm"),
    
    strip.placement = "outside",
    strip.background = element_rect(
      fill = "grey96",
      color = "grey55",
      linewidth = 0.35
    ),
    strip.text = element_text(
      size = 8.5,
      face = "bold",
      color = "grey10"
    ),
    
    panel.border = element_rect(
      color = "grey55",
      fill = NA,
      linewidth = 0.35
    ),
    panel.grid.major.y = element_line(
      color = "grey88",
      linewidth = 0.25
    ),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    
    panel.spacing.x = unit(0.55, "lines"),
    panel.spacing.y = unit(0.35, "lines"),
    
    axis.title = element_blank(),
    axis.text.x = element_text(
      size = 8,
      face = "bold",
      color = "grey20"
    ),
    axis.text.y = element_text(
      size = 7,
      color = "grey25"
    ),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_line(
      color = "grey45",
      linewidth = 0.25
    ),
    
    plot.margin = margin(4, 5, 4, 4)
  ) +
  guides(
    fill = guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(
        pattern = method_patterns,     
        pattern_angle = method_pattern_angle, 
        pattern_density = 0.15,         
        pattern_spacing = 0.03,         
        pattern_fill = "grey15",
        pattern_colour = "grey15"
      )
    ),
    pattern = "none",
    pattern_angle = "none"
  )

print(final_plot)
# ggsave(
#   filename = "sim_figure.pdf",
#   plot = final_plot,
#   width = 7.2,
#   height = 5.2
# )
