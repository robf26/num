
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


## OOB stacking ##
# adapted code from:
# https://www.kaggle.com/files/2676/DS06_nnet_Bagstack%20-%20Share.R
samples <- samplingcv("option2",trainindex,testindex,ytrain)
sampleindex <- lapply(samples,function(x) x$index)
index = sampleindex[[1]]
#data_f <- data_pca_all_scale_c[,c(1:10,22)]
data_f <- data
y <- data_f$target
data_f$target <- factor(data_f$target,labels = c("x0","x1"))

logloss.func <- function(x) -1*mean(log(ifelse(y==1,x,1-x)),na.rm=TRUE)

train_flag <- rep(FALSE,nrow(data_f))
train_flag[index$train] <- TRUE

###Put together the dataset of out of fold and final predictions for each base learner

library(foreach)
library(doParallel)
ptm <- proc.time()
cl <- makeCluster(detectCores()-1) 
registerDoParallel(cl)

bagged <- foreach(
  i=1:300
  #,.packages='nnet'
  #,.verbose=TRUE
) %dopar% {
  ##Bag 
  curr.train <- sample.int(sum(train_flag),sum(train_flag),TRUE)
  curr.oob <- c(!(seq(1,sum(train_flag)) %in% curr.train),rep(FALSE,sum(!train_flag)))
  ## randomise columns?
  ##Train 
  glm_fit <- glm(target~ .,
                 data=data_f[curr.train,],
                 family=binomial(logit))
  ##Pred 
  oob.pred <- rep(NA,sum(train_flag))
  oob.pred[curr.oob] <- predict(glm_fit,data_f[curr.oob,],type="response")
  return(list(
    oob.pred=oob.pred
    ,test.pred=predict(glm_fit,data_f[index$test,],type="response")
  ))
}

stopCluster(cl)
proc.time() - ptm

##Extract the pieces
res.oob.pred <- sapply(bagged,function(x) x$oob.pred)
res.list.oob.pred <- lapply(bagged,function(x) x$oob.pred)
res.test.pred <- sapply(bagged,function(x) x$test.pred)

##Show extremes
summary(apply(res.oob.pred,2,max,na.rm=TRUE));summary(apply(res.oob.pred,2,min,na.rm=TRUE))

##A single oob prediction for all
logloss.func(apply(res.oob.pred,1,mean,na.rm=TRUE))

##Logloss for each oob learner
res.oob.logloss <- apply(res.oob.pred,2,logloss.func)
summary(res.oob.logloss)












