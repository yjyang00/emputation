# scripts/config.R
# Central configuration for experiments (datasets, scenarios, hyperparams, paths)

cfg = list(
  # global
  seed_base = 1,
  m_imp = 10,          # number of multiple imputations
  n_iter = 100,        # number of iterations
  
  # data
  data_files = list(
    wine     = "data/winequality-white.csv",
    ccpp     = "data/Folds_pp.xlsx",
    concrete = "data/concrete_data.csv"
  ),
  
  xlsx_sheet = list(
    ccpp = NULL
  ),
  
  # Emputation hyperparams
  emputation_default = list(
    num_epochs = 500, 
    lr = 1e-4,
    M = 2,
    hidden_dim = 500,
    num_layer = 3,
    batch_norm = TRUE,
    force_full_batch = TRUE,
    mem_threshold_mb = Inf
  ),
  
  # GAIN
  gain = list(
    batch_size = 128,
    hint_rate  = 0.9,
    alpha      = 100,
    iterations = 1000
  ),
  
  # DGP
  ## MCAR (20%)
  mcar = list(
    concrete = list(rho = 0.25),
    ccpp     = list(rho = 0.25),
    wine     = list(rho = 0.25)
  ),
  
  # MAR (20%)
  mar = list(
    concrete = list(target_rate = 0.2),
    ccpp     = list(target_rate = 0.2),
    wine     = list(target_rate = 0.2)
  ),
  
  # scenario
  scenarios = list(
    list(dataset="concrete", dgp="mcar", train="mcar"),
    list(dataset="concrete", dgp="ccmv", train="ccmv"),
    list(dataset="concrete", dgp="mcar", train="ccmv"),
    list(dataset="concrete", dgp="ccmv", train="mcar"),
    list(dataset="concrete", dgp="mar", train="mcar"),
    list(dataset="concrete", dgp="mar", train="ccmv"),
    
    list(dataset="ccpp", dgp="mcar", train="mcar"),
    list(dataset="ccpp", dgp="ccmv", train="ccmv"),
    list(dataset="ccpp", dgp="mcar", train="ccmv"),
    list(dataset="ccpp", dgp="ccmv", train="mcar"),
    list(dataset="ccpp", dgp="mar", train="mcar"),
    list(dataset="ccpp", dgp="mar", train="ccmv"),
    
    list(dataset="wine", dgp="mcar", train="mcar"),
    list(dataset="wine", dgp="ccmv", train="ccmv"),
    list(dataset="wine", dgp="mcar", train="ccmv"),
    list(dataset="wine", dgp="ccmv", train="mcar"),
    list(dataset="wine", dgp="mar", train="mcar"),
    list(dataset="wine", dgp="mar", train="ccmv")
  )
)

scenario_tag = function(dataset, dgp, train) {
  paste0(dataset, "/dgp_", dgp, "/train_", train)
}

