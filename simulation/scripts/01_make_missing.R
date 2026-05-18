# scripts/01_make_missing.R
# Generate missing data (dat_NA) under DGP = mcar / mar / ccmv
# Save base artifact: iter_###.rda under run/<dataset>/dgp_<dgp>/train_<train>/

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
  stop("Usage: Rscript scripts/01_make_missing.R --scenario_id <int> --iter <int> [--overwrite 0/1] [--seed <int>]")
}

scenario_id = as.integer(opt$scenario_id)
iter        = as.integer(opt$iter)
overwrite   = if (!is.null(opt$overwrite)) as.integer(opt$overwrite) else 0

sc = get_scenario(scenario_id)
dataset = tolower(sc$dataset)
dgp     = tolower(sc$dgp)
train   = tolower(sc$train)

# seed
seed = if (!is.null(opt$seed)) as.integer(opt$seed) else (cfg$seed_base + iter)
set.seed(seed)

# output path
out_dir = run_scenario_dir(dataset, dgp, train)
ensure_dir(out_dir)
out_file = iter_base_rda(dataset, dgp, train, iter)
if (file.exists(out_file) && overwrite == 0) {
  cat("Base artifact exists; skipping (overwrite=0):", out_file, "\n")
  quit(save = "no", status = 0)
}

# load data
data_full = load_dataset(dataset)
d = ncol(data_full)

# MCAR
gen_mcar = function(dat, rho) {
  generate_mcar(dat, p = rho)
}

# MAR
gen_mar = function(dat, target_rate) {
  generate_mar(
    data = dat, 
    target_rate = target_rate
  )
}

# CCMV
gen_ccmv = function(dat, dataset) {
  spec = get_ccmv_spec(dataset, ncol(dat))
  dat_NA = ccmv(data = dat, patterns = spec$patterns, coeffs = spec$coeffs)
  list(dat_NA = dat_NA, spec = spec)
}

meta = list(
  scenario_id = scenario_id,
  dataset = dataset,
  dgp = dgp,
  train = train,
  iter = iter,
  seed = seed,
  created_at = as.character(Sys.time())
)

if (dgp == "mcar") {
  if (is.null(cfg$mcar[[dataset]]$rho)) stop("Missing cfg$mcar[[", dataset, "]]$rho")
  rho = cfg$mcar[[dataset]]$rho
  dat_NA = gen_mcar(data_full, rho = rho)
  meta$dgp_params = list(rho = rho)
  
} else if (dgp == "mar") {
  conf = cfg$mar[[dataset]]
  if (is.null(conf)) stop("Missing MAR config for: ", dataset)
  dat_NA = gen_mar(
    dat    = data_full, 
    target_rate   = conf$target_rate
  )
  meta$dgp_params = list(
    target_rate = conf$target_rate,
    actual_cell_miss = mean(is.na(dat_NA))
  )
  
} else if (dgp == "ccmv") {
  ccmv_obj = gen_ccmv(data_full, dataset = dataset)
  dat_NA = ccmv_obj$dat_NA
  meta$dgp_params = list(
    patterns_n = length(ccmv_obj$spec$patterns)
  )
} else {
  stop("Unsupported dgp: ", dgp, " (expected mcar/mar/ccmv)")
}

# sanity check
if (!all(dim(dat_NA) == dim(data_full))) {
  stop("Dimension mismatch: dat_NA dim = ", paste(dim(dat_NA), collapse="x"),
       " vs data_full dim = ", paste(dim(data_full), collapse="x"))
}

# save
save(data_full, dat_NA, iter, meta, file = out_file)
cat("Saved base artifact:) \n")
cat("  scenario:", scenario_tag(dataset, dgp, train), "\n")
cat("  iter:", iter, " seed:", seed, "\n")
cat("  file:", out_file, "\n")
