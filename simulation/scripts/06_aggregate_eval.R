# scripts/06_aggregate_eval.R
#
# Reads:
#   results/<dataset>/dgp_<dgp>/train_<train>/res_###.rda  
#
# Writes:
#   results/<dataset>/dgp_<dgp>/train_<train>/eval.rda    

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

if (is.null(opt$scenario_id)) {
  stop("Usage: Rscript scripts/06_aggregate_eval.R --scenario_id <int> [--overwrite 0/1]")
}

scenario_id = as.integer(opt$scenario_id)
overwrite   = if (!is.null(opt$overwrite)) as.integer(opt$overwrite) else 0

sc = get_scenario(scenario_id)
dataset = tolower(sc$dataset)
dgp     = tolower(sc$dgp)
train   = tolower(sc$train)

res_dir  = results_scenario_dir(dataset, dgp, train)
out_file = eval_agg_rda(dataset, dgp, train)

ensure_dir(res_dir)

if (file.exists(out_file) && overwrite == 0) {
  cat("Aggregate eval exists; skipping (overwrite=0):", out_file, "\n")
  quit(save = "no", status = 0)
}

files = list.files(res_dir, pattern = "^res_\\d+\\.rda$", full.names = TRUE)
files = sort(files)

if (length(files) == 0) {
  stop("No evaluation shard files found in: ", res_dir, "\nRun 05_eval_one_iter first.")
}

cat("Aggregating evaluation shards\n")
cat("  scenario:", scenario_tag(dataset, dgp, train), "\n")
cat("  dir:", res_dir, "\n")
cat("  shards found:", length(files), "\n")

# load
rows = list()
bad  = character()

for (f in files) {
  obj = tryCatch({
    e = new.env(parent = emptyenv())
    load(f, envir = e)
    if (!exists("row_res", envir = e)) stop("row_res not found")
    get("row_res", envir = e)
  }, error = function(err) {
    bad <<- c(bad, basename(f))
    NULL
  })
  
  if (!is.null(obj)) rows[[length(rows) + 1]] = obj
}

if (length(rows) == 0) {
  stop("All shards failed to load. Example failures: ", paste(head(bad, 5), collapse = ", "))
}

eval_df = dplyr::bind_rows(rows)

# sanity check
if (!("method" %in% names(eval_df))) stop("eval_df missing `method` column")
if (!("iter" %in% names(eval_df)))   stop("eval_df missing `iter` column")
metric_cols = intersect(c("RMSE", "MAE", "MADC", "Energy", "MMD2"), names(eval_df))
if (length(metric_cols) == 0) stop("No metric columns found in eval_df")

# summary
summary_df = eval_df |>
  dplyr::group_by(method) |>
  dplyr::summarise(
    dplyr::across(
      dplyr::all_of(metric_cols),
      list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  ) |>
  dplyr::select(method, dplyr::ends_with("_mean")) |>
  dplyr::arrange(method)

# save
meta_agg = list(
  scenario_id = scenario_id,
  dataset = dataset,
  dgp = dgp,
  train = train,
  res_dir = res_dir,
  n_shards = length(files),
  n_loaded = length(rows),
  failed_shards = bad,
  created_at = as.character(Sys.time())
)

save(eval_df, summary_df, meta_agg, file = out_file)

cat("Saved aggregate eval:)\n")
cat("  file:", out_file, "\n")