
library(data.table)
library(batchtools)
library(ggplot2)

set.seed(1042)

repls <- 100

# Registry ----------------------------------------------------------------
reg_name <- "rpf_sim_all"
reg_dir <- file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       packages = c("randomForest", "xgboost", "ranger"),
                       source = c("rpf.R", "predict_rpf.R", "ARCHIVE_Codes_used _in_paper/generate_data.R"))

# Problems -----------------------------------------------------------
myprob <- function(job, data, ...) {
  dat_train <- generate_data(rho=0.3,sparsity=2,sigma=1,  covariates='normal', ...)
  dat_test <- generate_data(rho=0.3,sparsity=2,sigma=1, covariates='normal', ...)
  
  list(train = dat_train, 
       test = dat_test)
}
addProblem(name = "myprob", fun = myprob, seed = 1043)

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
# Generate experiment design from best tuning parameters
best_params <- readRDS("best_params.Rds")
best_params <- best_params[order(algorithm, max_interaction, max.depth, Model, n, p), ]

# Problems
prob_design <- list(myprob = best_params[algorithm == "rf", .(n, p, Model)])

# Algorithms
algo_design <- list(
           rf = best_params[algorithm == "rf", .(ntree, mtry, maxnodes)], 
           xgboost = best_params[algorithm == "xgboost", .(eta, nrounds, max.depth)], 
           rpf = best_params[algorithm == "rpf", .(ntrees, splits, split_try, t_try, max_interaction)], 
           ranger = best_params[algorithm == "ranger", .(num.trees, mtry, max.depth, replace)])

addExperiments(prob_design, algo_design, repls = repls, combine = "bind")
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
saveRDS(res, "sim_result_all.Rds")

# Rename
res[, Method := factor(paste(algorithm, max.depth, max_interaction), 
                       levels = c("rf NA NA",  "xgboost 1 NA", "xgboost 2 NA", "xgboost 3 NA", "xgboost 4 NA", "rpf NA 1", "rpf NA 2", "rpf NA 3", "rpf NA 4", "ranger 1 NA", "ranger 2 NA", "ranger 3 NA", "ranger 4 NA", "ranger 0 NA"), 
                       labels = c("RF", "xgboost additive", "xgboost interaction 2", "xgboost interaction 3",  "xgboost interaction 4", "RPF additive", "RPF interaction 2", "RPF interaction 3", "RPF interaction 4", "ranger additive", "ranger interaction 2", "ranger interaction 3", "ranger interaction 4", "ranger"))]

# Average over repls
res_mean <- res[, mean(mse), by = .(problem, algorithm, Model, Method, n, p, ntree, mtry, maxnodes, eta, nrounds, max.depth, ntrees, splits, split_try, t_try, max_interaction, num.trees, replace)]
res_mean[, mse := V1]
res_mean
res_mean[Model == 2 & p == 4, .(Method, round(V1,3))]

# Plot results -------------------------------------------------------------
res_plot <- res[!(Method %in% c("ranger additive", "ranger interaction 2", "ranger interaction 3", "ranger interaction 4", "ranger")), ]
ggplot(res_plot, aes(x = Method, y = mse)) +
  geom_boxplot() + 
  coord_flip() + 
  facet_grid(p ~ Model, scales = "free") + 
  ylab("MSE")
ggsave("sim_all.pdf", width = 15, height = 10)

# Export results -------------------------------------------------------------
res_export <- copy(res)
res_export[, problem := NULL]
res_export[, result.1 := NULL]
saveRDS(res_export, "sim_export.Rds")
