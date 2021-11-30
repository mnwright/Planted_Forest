
library(data.table)
library(batchtools)
library(ggplot2)

set.seed(42)

repls <- 30

# Data
n <- 500
p <- c(4, 10, 30)
Model <- 1:6

# RF
ntree <- 500
mtry <- c(1/4, 1/2, 3/4) # will be multiplied by p
maxnodes <- c(40, 60, 80, 100, 120, n)

# xgboost
eta <- c(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
nrounds <- c(100, 300, 600, 1000, 3000, 5000, 7000)
max.depth <- c(1, 2, 3, 4)

# RPF
ntrees <- 50
splits <-  c(10, 15, 20, 25, 30, 30, 40, 50, 60, 80, 100, 120)
split_try <- c(2, 5, 10, 20, 5, 20)
t_try <- c(0.25, 0.5, 0.75)
max_interaction <- c(1, 2, 3, 4)

# ranger
num.trees <- 500
mtry_rg <- c(1/4, 1/2, 3/4, 7/8, 1) # will be multiplied by p
max.depth_rg <- c(1, 2, 3, 4, 0)
replace <- c(TRUE, FALSE)

# Registry ----------------------------------------------------------------
reg_name <- "rpf_tune_all"
reg_dir <- file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       packages = c("randomForest", "xgboost", "ranger"),
                       source = c("rpf.R", "predict_rpf.R", "ARCHIVE_Codes_used _in_paper/generate_data.R"))

# Problems -----------------------------------------------------------
myprob <- function(job, data, ...) {
  dat_train <- generate_data(rho=0.3,sparsity=2,sigma=1, covariates='normal', ...)
  dat_test <- generate_data(rho=0.3,sparsity=2,sigma=1, covariates='normal', ...)
  
  list(train = dat_train, 
       test = dat_test)
}
addProblem(name = "myprob", fun = myprob, seed = 43)

# Algorithms -----------------------------------------------------------
run_rf <- function(data, job, instance, mtry, ...) {
    fit <- randomForest(x = instance$train$X,
                        y = instance$train$Y_start,
                        mtry = mtry * ncol(instance$train$X), 
                        ...)
    pred <- predict(fit, instance$test$X)
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "rf", fun = run_rf)

run_xgboost <- function(data, job, instance, ...) {
    fit <- xgboost(data = instance$train$X,
                   label = instance$train$Y_start,
                   nthread = 1,
                   early_stopping_rounds = NULL,
                   verbose = F,
                   objective = "reg:squarederror", 
                   ...)
    pred <- predict(fit, instance$test$X)
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "xgboost", fun = run_xgboost)

run_rpf <- function(data, job, instance, ...) {
    fit <- rpf(X=instance$train$X,
               Y=instance$train$Y_start,
               variables=NULL,
               min_leaf_size=1, 
               ...)
    pred <- predict_rpf(forest_res = fit, X = instance$test$X)
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "rpf", fun = run_rpf)

run_ranger <- function(data, job, instance, mtry, ...) {
    colnames(instance$train$X) <- paste0("X", 1:ncol(instance$train$X))
    colnames(instance$test$X) <- paste0("X", 1:ncol(instance$test$X))
    fit <- ranger(x = instance$train$X,
                  y = instance$train$Y_start,
                  mtry = mtry * ncol(instance$train$X), 
                  ...)
    pred <- predict(fit, instance$test$X)$predictions
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "ranger", fun = run_ranger)

# Experiments -----------------------------------------------------------
prob_design <- list(myprob = expand.grid(n = n, 
                                         p = p, 
                                         Model = Model,
                                         stringsAsFactors = FALSE))
algo_design <- list(rf = expand.grid(ntree = ntree, 
                                     mtry = mtry, 
                                     maxnodes = maxnodes,
                                     stringsAsFactors = FALSE), 
                    xgboost = expand.grid(eta = eta,
                                          nrounds = nrounds, 
                                          max.depth = max.depth, 
                                          stringsAsFactors = FALSE), 
                    rpf = expand.grid(ntrees = ntrees, 
                                      splits = splits,
                                      split_try = split_try, 
                                      t_try = t_try,
                                      max_interaction = max_interaction,
                                      stringsAsFactors = FALSE), 
                    ranger = expand.grid(num.trees = num.trees, 
                                         mtry = mtry_rg, 
                                         max.depth = max.depth_rg,
                                         replace = replace,
                                         stringsAsFactors = FALSE))
addExperiments(prob_design, algo_design, repls = repls)
summarizeExperiments()

# Test jobs -----------------------------------------------------------
#testJob(id = 1)
#testJob(id = 500)

# Submit -----------------------------------------------------------
if (grepl("node\\d{2}|bipscluster", system("hostname", intern = TRUE))) {
  ids <- findNotStarted()
  ids[, chunk := chunk(job.id, chunk.size = 50)]
  submitJobs(ids = ids, # walltime in seconds, 10 days max, memory in MB
             resources = list(name = reg_name, chunks.as.arrayjobs = TRUE, 
                              ncpus = 1, memory = 6000, walltime = 10*24*3600, 
                              max.concurrent.jobs = 400))
} else {
  submitJobs()
}
waitForJobs()

# Get results -------------------------------------------------------------
res <-  flatten(ijoin(reduceResultsDataTable(), getJobPars()))
res[, mse := result.1]

# Save result
saveRDS(res, "tune_result_all.Rds")

# Average over repls
res_mean <- res[, mean(mse), by = .(problem, algorithm, n, p, Model, ntree, mtry, maxnodes, eta, nrounds, max.depth, ntrees, splits, split_try, t_try, max_interaction, num.trees, replace)]
res_mean[, mse := V1]

# Get best parameters per method
best_params <- res_mean[ , .SD[which.min(mse)], by = .(algorithm, n, p, Model, max_interaction, max.depth)]
saveRDS(best_params, "best_params.Rds")

# Results
# RF: 
# xgboost additive (max.depth=2):
# xgboost interaction: 
# RPF additive: 
# RPF interaction: 
