# scripts/04_missforest.R
# missForest imputation (default parameters) per iteration.
#
# Reads:
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###.rda
#
# Writes:
#   run/<dataset>/dgp_<dgp>/train_<train>/iter_###_missForest.rda

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
  stop("Usage: Rscript scripts/04_missforest.R --scenario_id <int> --iter <int> [--overwrite 0/1]")
}

scenario_id = as.integer(opt$scenario_id)
iter        = as.integer(opt$iter)
overwrite   = if (!is.null(opt$overwrite)) as.integer(opt$overwrite) else 0

sc = get_scenario(scenario_id)
dataset = tolower(sc$dataset)
dgp     = tolower(sc$dgp)
train   = tolower(sc$train)

# path
base_file = iter_base_rda(dataset, dgp, train, iter)
out_file  = iter_missforest_rda(dataset, dgp, train, iter)

ensure_dir(run_scenario_dir(dataset, dgp, train))

if (!file.exists(base_file)) {
  stop("Base artifact not found. Run 01_make_missing first.\n  expected: ", base_file)
}

if (file.exists(out_file) && overwrite == 0) {
  cat("missForest exists; skipping (overwrite=0):", out_file, "\n")
  quit(save = "no", status = 0)
}


# load
load(base_file) 
if (!exists("dat_NA")) stop("dat_NA not found in base artifact")
dat_NA = as.data.frame(dat_NA)


# run missForest
cat("Running missForest\n")
cat("  scenario:", scenario_tag(dataset, dgp, train), "\n")
cat("  iter:", iter, "\n")

set.seed(cfg$seed_base + iter)
mf_fit = missForest(dat_NA)
mf_imp = as.data.frame(mf_fit$ximp)

# save
meta_mf = list(
  scenario_id = scenario_id,
  dataset = dataset,
  dgp = dgp,
  train = train,
  iter = iter,
  seed = cfg$seed_base + iter,
  base_file = base_file,
  created_at = as.character(Sys.time())
)

save(mf_imp, meta_mf, file = out_file)
cat("Saved missForest :)\n")
cat("  file:", out_file, "\n")

