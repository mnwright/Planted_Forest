
library(ranger)
library(ggplot2)
library(data.table)

n <- 100
p <- 4
repls <- 30

# y independent of x and variance 1
# It has to be impossible to achieve MSE < 1 (on average)
sim_data <- function(n, p) {
  x <- data.frame(replicate(p, rnorm(n)))
  y_true <- rnorm(n, 0, 1)
  y_start <- y_true + rnorm(n, 0, 1)
  list(x = x, y_true = y_true, y_start = y_start)
}

# Check with different evaluation methods
res <- replicate(repls, {
  dat_train <- sim_data(n, p)
  dat_test <- sim_data(n, p)
  
  # Fit RF and evaluate on out-of-sample data
  fit1 <- ranger(y = dat_train$y_true, x = dat_train$x)
  pred1 <- predict(fit1, dat_test$x)$predictions
  mse1 <- mean((dat_test$y_true - pred1)^2)
  
  # Fit RF on data with additional noise and evaluate against true y
  fit2 <- ranger(y = dat_train$y_start, x = dat_train$x)
  pred2 <- predict(fit2, dat_train$x)$predictions
  mse2 <- mean((dat_train$y_true - pred2)^2)
  
  # Sanity check: Just use training data
  fit3 <- ranger(y = dat_train$y_true, x = dat_train$x)
  pred3 <- predict(fit3, dat_train$x)$predictions
  mse3 <- mean((dat_train$y_true - pred3)^2)
  
  # Return all MSEs
  c(out_of_sample = mse1, 
    in_sample_noise = mse2, 
    in_sample = mse3)
})

# Plot
dt <- melt(as.data.table(t(res)), 
           measure.vars = c("out_of_sample", "in_sample_noise", "in_sample"))
ggplot(dt, aes(x = variable, y = value)) + 
  geom_boxplot() + 
  geom_hline(yintercept = 1, col = "red") + 
  xlab("") + ylab("MSE")
ggsave("eval_check.pdf")

# Results:
# Evaluation in sample gives a heavily biased estimate of prediction performance (obviously)
# Evaluation in sample with additional noise to training also gives a biased estimate of prediction performance
# Evaluation out of sample is fine (overfitting probably leads to MSE > 1)
