
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
  
  models_ens <- list()
  
  # For each samples, loop
  for (s in 1:length(prob_valid)) {
    
    model <- list()
    x <- prob_valid
    
    
    
    model$prob$valid <- NA
    
    
    # Predict on test set?
    model$prob$test <- NA
    
    # Add to model
    models_ens <- c(models_ens,list(model))
  }
  
  return(models_ens)
}


# Code to optimise weights on validation set. 



old <- function(x) {
  

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

# Although not really necessary here, fit a greedy optimisation on weights
set.seed(1)
X <- as.matrix(PredAll[PredAll$DataType=="Validation",modelnames])
Y <- PredAll[PredAll$DataType=="Validation","Y"]

iter        <- 150L
N           <- ncol(X)
weights     <- rep(0L, N)
pred        <- 0 * X
sum.weights <- 0L
maxtest     <- rep(0,iter)

while(sum.weights < iter) {
  sum.weights   <- sum.weights + 1L
  pred          <- (pred + X) * (1L / sum.weights) 
  # errors        <- sqrt(colSums((pred - Y) ^ 2L, na.rm=TRUE))  # RMSE
  errors        <- colSums(ifelse(pred==Y,0,1))  # check?
  best          <- which.min(errors)
  weights[best] <- weights[best] + 1L
  pred          <- pred[, best] * sum.weights
  maxtest[sum.weights] <- min(errors)
}
weights2 <- weights/sum(weights)

rbind(modelnames,weights2)

# Given the small validation set, and how correlated the predictors are,
# this doesn't really seem to be a good option


}








