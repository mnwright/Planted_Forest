
library(data.table)
library(batchtools)
library(ggplot2)

set.seed(1042)

n <- 500
p <- 30
repls <- 100

# RF
ntree <- 500
mtry <- 22
maxnodes <- 500

# xgboost
eta <- c(0.04, 0.02)
nrounds <- c(7000, 7000)
max.depth <- c(2, 3)

# RPF
ntrees <- c(50, 50)
splits <- c(30, 30)
split_try <- c(20, 20)
t_try <- c(0.75, 0.75)
max_interaction <- c(1, 2)

# Registry ----------------------------------------------------------------
reg_name <- "rpf_sim"
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
addProblem(name = "myprob", fun = myprob, seed = 1043)

# Algorithms -----------------------------------------------------------
run_rf <- function(data, job, instance, ...) {
    fit <- randomForest(x = instance$train$X,
                        y = instance$train$Y_true,
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
    fit <- rpf(X = instance$train$X,
               Y = instance$train$Y_true,
               variables = NULL,
               min_leaf_size = 1, 
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
algo_design <- list(rf = data.frame(ntree = ntree, 
                                     mtry = mtry, 
                                     maxnodes = maxnodes,
                                     stringsAsFactors = FALSE), 
                    xgboost = data.frame(eta = eta,
                                          nrounds = nrounds, 
                                          max.depth = max.depth, 
                                          stringsAsFactors = FALSE), 
                    rpf = data.frame(ntrees = ntrees, 
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
saveRDS(res, "sim_result.Rds")

# Average over repls
res_mean <- res[, mean(mse), by = .(problem, algorithm, n, p, ntree, mtry, maxnodes, eta, nrounds, max.depth, ntrees, splits, split_try, t_try, max_interaction)]
res_mean[, mse := V1]
res_mean

# Plot results -------------------------------------------------------------
res[, Method := factor(paste(algorithm, max.depth, max_interaction), 
                       levels = c("rf NA NA", "xgboost 2 NA", "xgboost 3 NA", "rpf NA 1", "rpf NA 2"), 
                       labels = c("RF", "xgboost additive", "xgboost interaction", "RPF additive", "RPF interaction"))]
ggplot(res, aes(x = Method, y = mse)) +
  geom_boxplot() + 
  coord_flip() + 
  ylab("MSE") + 
  theme_bw()
ggsave("model4_p30.pdf", width = 10, height = 8)

