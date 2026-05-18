# scripts/02_run_emputation.R
# Train Emputation and do multiple imputations.
#
# Reads:
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###.rda
# Writes:
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###_emputation.rda

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
  stop("Usage: Rscript scripts/02_run_emputation.R --scenario_id <int> --iter <int> [--overwrite 0/1] [--seed <int>]")
}

scenario_id = as.integer(opt$scenario_id)
iter        = as.integer(opt$iter)
overwrite   = if (!is.null(opt$overwrite)) as.integer(opt$overwrite) else 0
force_full_batch_override = if (!is.null(opt$force_full_batch)) as.integer(opt$force_full_batch) else NULL

sc = get_scenario(scenario_id)
dataset = tolower(sc$dataset)
dgp     = tolower(sc$dgp)
train   = tolower(sc$train)
seed = if (!is.null(opt$seed)) as.integer(opt$seed) else (cfg$seed_base + 100000 * scenario_id + iter)
set.seed(seed)

# path
base_file = iter_base_rda(dataset, dgp, train, iter)
out_file  = iter_emputation_rda(dataset, dgp, train, iter)
ensure_dir(run_scenario_dir(dataset, dgp, train))

if (!file.exists(base_file)) {
  stop("Base artifact not found. Run 01_make_missing first.\n  expected: ", base_file)
}

if (file.exists(out_file) && overwrite == 0) {
  cat("Emputation artifact exists; skipping (overwrite=0):", out_file, "\n")
  quit(save = "no", status = 0)
}

# load
load(base_file)
if (!exists("dat_NA")) stop("dat_NA not found in base artifact: ", base_file)
if (!exists("meta"))   stop("meta not found in base artifact: ", base_file)


# train emputation
params = get_emputation_params(dataset)
if (!is.null(force_full_batch_override)) {
  params$force_full_batch <- as.logical(force_full_batch_override)
  params$batch_size       <- NULL
  message("Override: force_full_batch set to ", params$force_full_batch, " via command line")
}
req = c("num_epochs", "lr", "M", "hidden_dim", "num_layer")
cat("Running Emputation\n")
cat("  scenario:", scenario_tag(dataset, dgp, train), "\n")
cat("  iter:", iter, "\n")
cat("  train mechanism:", train, "\n")
cat("  seed:", seed, "\n")
cat("  epochs:", params$num_epochs, " lr:", params$lr, " M:", params$M,
    " hidden_dim:", params$hidden_dim, " num_layer:", params$num_layer, 
    " batch_norm:", params$batch_norm, " force_full_batch:", params$force_full_batch,
    " mem_threshold_mb:", params$mem_threshold_mb, " batch_size:", params$batch_size, "\n")

# fit
emputationFit = emputation(
  dat_NA,
  mechanism  = train,
  num_epochs = params$num_epochs,
  lr         = params$lr,
  M          = params$M,
  hidden_dim = params$hidden_dim,
  num_layer  = params$num_layer,
  batch_norm = params$batch_norm,
  force_full_batch = params$force_full_batch,
  mem_threshold_mb = params$mem_threshold_mb,
  batch_size       = params$batch_size
)

# MI
m_imp = cfg$m_imp
emp_complete = predict(emputationFit, dat_NA, m = m_imp)

# save
meta_empu = list(
  scenario_id = scenario_id,
  dataset = dataset,
  dgp = dgp,
  train = train,
  iter = iter,
  seed = seed,
  base_file = base_file,
  created_at = as.character(Sys.time())
)

save(emp_complete, params, meta_empu, file = out_file)
cat("Saved emputation artifact:) \n")
cat("  file:", out_file, "\n")

