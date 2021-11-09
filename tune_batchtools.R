
library(data.table)
library(batchtools)
library(ggplot2)

set.seed(42)

n <- 500
p <- 30
repls <- 30

# RF
ntree <- 500
mtry <- c(floor(p/4), floor(p/2), floor(3*p/4))
maxnodes <- c(40, 60, 80, 100, 120, n)

# xgboost
eta <- c(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
nrounds <- c(100, 300, 600, 1000, 3000, 5000, 7000)
max.depth <- c(2, 3, 4)

# RPF
ntrees <- 50
splits <- c(10, 15, 20, 25, 30)
split_try <- c(2, 5, 10, 20)
t_try <- c(0.25, 0.5, 0.75)
max_interaction <- c(1, 2)

# Registry ----------------------------------------------------------------
reg_name <- "rpf_tune"
reg_dir <- file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       packages = c("randomForest", "xgboost"),
                       source = c("rpf.R", "predict_rpf.R", "ARCHIVE_Codes_used _in_paper/generate_data.R"))

# Problems -----------------------------------------------------------
myprob <- function(job, data, ...) {
  dat_train <- generate_data(rho=0.3,sparsity=2,sigma=1, Model=4, covariates='normal', ...)
  dat_test <- generate_data(rho=0.3,sparsity=2,sigma=1, Model=4, covariates='normal', ...)
  
  list(train = dat_train, 
       test = dat_test)
}
addProblem(name = "myprob", fun = myprob, seed = 43)

# Algorithms -----------------------------------------------------------
run_rf <- function(data, job, instance, ...) {
    fit <- randomForest(x=instance$train$X,
                        y=instance$train$Y_true,
                        ...)
    pred <- predict(fit, instance$test$X)
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "rf", fun = run_rf)

run_xgboost <- function(data, job, instance, ...) {
    fit <- xgboost(data = instance$train$X,
                   label = instance$train$Y_true,
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
               Y=instance$train$Y_true,
               variables=NULL,
               min_leaf_size=1, 
               ...)
    pred <- predict_rpf(forest_res = fit, X = instance$test$X)
    mse <- mean((pred-instance$test$Y_true)^2)
    mse
}
addAlgorithm(name = "rpf", fun = run_rpf)


# Experiments -----------------------------------------------------------
prob_design <- list(myprob = expand.grid(n = n, 
                                         p = p, 
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
saveRDS(res, "tune_result.Rds")

# Average over repls
res_mean <- res[, mean(mse), by = .(problem, algorithm, n, p, ntree, mtry, maxnodes, eta, nrounds, max.depth, ntrees, splits, split_try, t_try, max_interaction)]
res_mean[, mse := V1]

# Get best parameters per method
res_mean[ , .SD[which.min(mse)], by = .(algorithm, n, p, max_interaction, max.depth)]

# Results
# RF: mtry=22, maxnodes=500
# xgboost additive (max.depth=2): eta=0.04, nrounds=7000
# xgboost interaction: max.depth=3, eta=0.02, nrounds = 7000
# RPF additive: splits=30, split_try=20, t_try=.075
# RPF interaction: splits=30, split_try=20, t_try=.075
