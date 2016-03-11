
# ensemble.R
# Scripts to ensemble models.

ensemble_average_pred <- function(pred,type) {
  model <- list()
  for (s in 1:length(pred)) {
    fit <- list()
    fit$name = "ensemble_average"
    prob <- list()
    fit$prob[[type]] <- rowMeans(pred[[s]])
    model <- c(model,list(fit))
  }
  return(model)
}

fitallensembles <- function(models,samples_validindex_y,
                            samples_testindex_y) {
  
  prob_valid <- extract_pred_as_samples_models_df(models,"valid") 
  prob_test <- extract_pred_as_samples_models_df(models,"test") 
  
  models_n <- length(models)
  models_ens <- list()
  
  ## Fit ensemble equal weight ##
  fit <- list()
  samples_n <- length(prob_valid)
  for (s in 1:samples_n) {
    model <- list()
    model$name <- "ens_average"
    weights <- rep(1,models_n)/models_n
    model$param$weights <- weights
    prob <- list()
    prob$valid <- as.matrix(prob_valid[[s]]) %*% weights
    prob$test <- as.matrix(prob_test[[s]]) %*% weights
    model$prob <- prob
    fit <- c(fit,list(model))
  }
  models_ens <- c(models_ens,list(fit))
  #####
  
  ## Fit greedy ensemble ##
  fit <- list()
  samples_n <- length(prob_valid)
  for (s in 1:samples_n) {
    model <- list()
    model$name <- "ens_greedy_ll"
    x <- as.matrix(prob_valid[[s]])
    y <- samples_validindex_y[[s]]
    weights <- greedylogloss(x,y)
    model$param$weights <- weights
    prob <- list()
    prob$valid <- x %*% weights
    prob$test <- as.matrix(prob_test[[s]]) %*% weights
    model$prob <- prob
    fit <- c(fit,list(model))
  }
  models_ens <- c(models_ens,list(fit))
  #####
  
  return(models_ens)
}


# Greedy optimisation
greedylogloss <- function(x,y) {
  # x: probabilities
  # y: actual outcome
  iter        <- 100L
  N           <- ncol(x)
  weights     <- rep(0L, N)
  pred        <- 0 * x
  sum.weights <- 0L
  maxtest     <- rep(0,iter)
  
  while(sum.weights < iter) {
    sum.weights   <- sum.weights + 1L
    pred          <- (pred + x) * (1L / sum.weights) 
    # errors        <- sqrt(colSums((pred - Y) ^ 2L, na.rm=TRUE))  # RMSE
    # errors        <- colSums(ifelse(pred==Y,0,1))  # 
    errors <- apply(pred,2,function(z) logLoss(y,z))  # apply logloss option
    best          <- which.min(errors)
    weights[best] <- weights[best] + 1L
    pred          <- pred[, best] * sum.weights
    maxtest[sum.weights] <- min(errors)
  }
  weightsnorm <- weights/sum(weights)
  return(weightsnorm)
}




WAVE <- function(x) {
  
# Implement the WAVE algorithm for ensembling
# http://www.ams.sunysb.edu/~hahn/psfile/wave.pdf
# Also implement a greedy optimisation

str(PredAll)
modelnames
# We want to optimise on the validation set
unique(PredAll$DataType)
X <- as.matrix(PredAll[PredAll$DataType=="Validation",paste0("correct.",modelnames)])
# Code so that 1 represents a correct decision, 0 is incorrect

apply(X,2,mean)

n <- nrow(X)
k <- ncol(X)
Jnk <- matrix(rep(1,n*k,),n,k)
Jkk <- matrix(rep(1,k*k,),k,k)
Ik <- diag(k)
onek <- rep(1,k)
onen <- rep(1,n)
# Iterative approach
Q <- ((Jnk - X) %*% (Jkk - Ik) %*% onek) / as.numeric((t(onen) %*% (Jnk - X) %*% (Jkk - Ik) %*% onek))
iter <- 100
# for loop 1 to m
for (m in 1:iter) {
  P <- t(X) %*% Q / as.numeric(onek %*% t(X) %*% Q)
  Q <- ((Jnk - X) %*% (Jkk - Ik) %*% P) / as.numeric((t(onen) %*% (Jnk - X) %*% (Jkk - Ik) %*% P))
}
# Weights for each classifier
P
# Importance of each instance contributing to weights
Q

# Individual predictions
testPred <- PredAll[PredAll$DataType=="Test",modelnames]
testPredProb <- PredAll[PredAll$DataType=="Test",paste0("prob.",modelnames)]

# Ensemble predictions
ensemblePred <- ifelse(as.matrix(testPred) %*% P>=0,1,-1)
ensemblePredProb <- ifelse(as.matrix(testPredProb) %*% P>=0.5,1,-1)

EnsembleResults <- data.frame(ensemblePred,ensemblePredProb,testPred)
EnsembleResults
cor(EnsembleResults)

propcorrect <- ifelse(EnsembleResults==PredAll[PredAll$DataType=="Test","Y"],1,0)
apply(propcorrect,2,mean)
P

}








