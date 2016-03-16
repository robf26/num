
# models.R
# model functions:
# Take an index list including train, and optional valid, test
# returns the model object with risk metrics, and optional prediction probabilities.
saveprob = TRUE

fitallmodels <- function(data,samples,saveprob = TRUE) {
  # FIX: Include option to return model_samples or samples_model
  # Option to probabilities to csv file (if exceeding memory requirements?)
  
  models <- list()
  
  sampleindex <- lapply(samples,function(x) x$index)
  
  # Fit glm 
  model <- lapply(sampleindex,function(x) fitglmsub(x))
  models <- c(models,list(model))
  print("Finished fitting glm")
  
  # Fit other models based on cluster pairs
  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust1",clust1))
  models <- c(models,list(model))
  print("Finished fitting glm_clust1")
  
  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust2",clust2))
  models <- c(models,list(model))
  print("Finished fitting glm_clust2")
  
  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust3",clust3))
  models <- c(models,list(model))
  print("Finished fitting glm_clust3")
  
  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust4",clust4))
  models <- c(models,list(model))
  print("Finished fitting glm_clust4")
  
  # glm net, elasto net
  model <- lapply(sampleindex,function(x) fitglmnet(x))
  models <- c(models,list(model))
  print("Finished fitting glmnet")
  
  # fit nnet
  # glm net, elasto net
  model <- lapply(sampleindex,function(x) fitnnet(x))
  models <- c(models,list(model))
  print("Finished fitting nnet")
  
  return(models)
}

## glm with ability to fit on certain features only ##
fitglmsub <- function(index,
                      name="glm",features="all") {

  model <- list()
  model$name = name
  if (as.vector(features)[1]=="all") {
    features = names(data)
  } else {
    features = c("target",features)
  }
  model_fit <- glm(target~ .,
                   data=data[index$train,features],
                   family=binomial(logit))
  prob <- lapply(index,function(x) {predict(model_fit,
                                            newdata=data[x,features],
                                            type="response")})

  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index, prob, SIMPLIFY = FALSE)
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  #models <- append(models,list(model))
  return(model)
}


fitglmnet <- function(index,
                      name="glmnet",features="all") {
  
  #require(glmnet)
  #samples <- samplingcv("option2",trainindex,testindex,ytrain)
  #sampleindex <- lapply(samples,function(x) x$index)
  #index = sampleindex[[1]]
  
  model <- list()
  model$name = name
  if (as.vector(features)[1]=="all") {
    features = names(data)[-length(names(data))]
  }
  model_fit <- glmnet(as.matrix(data[index$train,features]),
                      data[index$train,"target"],
                      family="binomial")
  #plot(model_fit,label=TRUE)
  #plot(model_fit, xvar = "dev", label = TRUE)
  #cvfit = cv.glmnet(as.matrix(data[index$train,features]), 
  #                  data[index$train,"target"], 
  #                  family = "binomial", type.measure = "auc")
  #plot(cvfit)
  #cvfit$lambda.min
  
  prob <- lapply(index,function(x) {predict(model_fit,
                                            newx=as.matrix(data[x,features]),
                                            type="response",s=0.002)})
  
  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index, prob, SIMPLIFY = FALSE)
  
  # Save output
  #if (saveprob) {
    model$prob <- prob
  #}
  model$logloss <- logloss
  #models <- append(models,list(model))
  return(model)
}

# Fit a basic neural network
fitnnet <- function(index,
                      name="nnet") {
  #samples <- samplingcv("option2",trainindex,testindex,ytrain)
  #sampleindex <- lapply(samples,function(x) x$index)
  #index = sampleindex[[1]]
  
  model <- list()
  model$name = name
  require(nnet)
  yTrain1 <- data.frame(all)[,outcome_name]
  yNet1 <- class.ind(yTrain1)
  model_fit <- nnet(data[index$train,clust4],yNet1[index$train,],
                size=9,maxit = 100,softmax=TRUE)

  prob <- lapply(index,function(x) 
            {predict(model_fit,data[x,clust4])[,2]})
  
  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index, prob, SIMPLIFY = FALSE)
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  return(model)
}


## Model that runs k-means then runs pca then fits glmnet.


## Other models
old <- function(x) {
  
  # Now lets try bagging glm
  # bagging logistic regression
  # http://stackoverflow.com/questions/21785699/bagging-logistic-regression-in-r
  
  sampleindex <- lapply(samples,function(x) x$index)
  
  library(foreach)
  set.seed(1)
  training2      <- data.frame(all)[trainindex,1:22]
  length_divisor <- 2
  iterations     <- 50
  features <- 10
  predictions <- foreach(m=1:iterations,.combine=rbind) %do% {
    training_positions <- sample(nrow(training2), 
                                 size=floor((nrow(training2)/length_divisor)))
    #  train_pos<-1:nrow(training2) %in% training_positions
    glm_fit <- glm(target~ . ,
                   data=training2[training_positions,
                                  c(sample(1:21,features),22)],
                   family=binomial(logit))
    predict(glm_fit,
            newdata=xTest, #xTest
            type="response")
  }
  #predictions
  bag_prob <- apply(t(predictions),1,mean)
  logLoss(actual_train, bag_prob)
  
  cor(data.frame(bag_prob,models_prob(models,xTrain)))
  cor(data.frame(bag_prob,models_prob(models,xTest)))
  
  ensemble_prob_test <- apply(data.frame(bag_prob,
                                         models_prob(models,xTest)),
                              1,mean)
  
  submission <- data.frame(t_id = all[testindex,"t_id"],
                           probability = ensemble_prob_test)
  
  # add in proper cross validation on above.
  
  # Check individual factor signs
  coeff <- NA
  for (i in 1:21) {
    fit <- glm(target~.,binomial,
               all[trainindex,c(i,22)])
    summary(fit)
    coeff[i] <- fit$coefficients[2]
  }
  signs <- sign(coeff)
  
  # Set up validation set
  set.seed(1)
  trainindex1 <- sample(trainindex,floor(0.7*length(trainindex)))
  trainindex2 <- setdiff(trainindex,trainindex1)
  require(gbm)
  set.seed(5)
  mgbm1 <- gbm(target~.,
               "bernoulli",
               data=all[trainindex1,c(outcome_name,feature_names)],
               var.monotone=signs,
               n.trees=50,
               interaction.depth=1,
               n.minobsinnode = 1,
               shrinkage = 0.05,
               bag.fraction = 0.5,
               train.fraction = 0.5,
               n.cores = 7)
  
  probgbm = predict(mgbm1,all[trainindex2,],type="response") 
  actualv <- data.frame(all)[trainindex2,outcome_name]
  logLoss(actualv, probgbm)
  
  probgbm <- predict(mgbm1,xTrain,type="response") 
  
  cor(data.frame(probgbm,models_prob(models,xTrain)))
  
  probgbm <- predict(mgbm1,xTest,type="response") 
  
  ensemble_prob_test <- apply(data.frame(probgbm,bag_prob,
                                         models_prob(models,xTest)),
                              1,mean)
  
  submission <- data.frame(t_id = all[testindex,"t_id"],
                           probability = ensemble_prob_test)
  
  write_csv(submission,"./predictions/predictions_9.csv")
  
  # Maybe look to remove the gbm. Or learn to fit better?
  
  # Next try SVMs. Although may need to train on smaller sample size?
  # takes foreever to train!
  require(kernlab)
  datat <- data.frame(target=yTrainExValid,xTrainExValid)
  
  # ok with 5000, slows down with >10K
  svm1 <- ksvm(target~.,data=datat[1:10000,c("target",clust4)],
               kernel="rbfdot",
               kpar=list(sigma=0.1),C=1,
               prob.model=TRUE)
  svm_prob <- predict(svm1,xValid[,clust4], type="probabilities")
  logLoss(actualv,svm_prob)
  # Still pretty rubbish fit
  # Tube with other costs and sigma?
  svm2 <- ksvm(target~.,data=datat[30001:40000,c("target",clust4)],
               kernel="rbfdot",C=0.5,
               prob.model=TRUE)
  svm_prob <- predict(svm2,xValid, type="probabilities")[,2]
  logLoss(actualv,svm_prob)
  
  #svm_prob1 <- svm_prob
  #svm_prob2 <- svm_prob
  svm_prob3 <- svm_prob
  cor(data.frame(svm_prob1,svm_prob2,svm_prob3))
  
  cor(data.frame(svm_prob,models_prob(models,xValid)))
  
  library(doParallel)
  cl <- makeCluster(detectCores()-1) 
  registerDoParallel(cl)
  Train <- data.frame(target=yTrain,xTrain)
  trcontrol_2 = trainControl(
    method = "repeatedcv",
    number = 3,
    repeats = 3,
    returnResamp= "none",
    verboseIter = TRUE,
    returnData = FALSE,
    classProbs = TRUE, 
    summaryFunction = multiClassSummary,
    allowParallel = TRUE
  )
  set.seed(1)
  svm_grid <- expand.grid(sigma = c(.015),C = c(0.25))
  svm_sample_size = 3000
  svm_sample_index <- sample(1:length(yTrain),svm_sample_size)
  svm3 <- train(target ~ ., data=Train[svm_sample_index,
                                       c("target",clust1)], 
                method="svmRadial",
                trControl = trcontrol_2,
                tuneGrid = svm_grid,
                metric = "logLoss")
  svm_prob <- predict(svm3,xValid[,clust1], type="prob")[,2]
  logLoss(actualv,svm_prob)
  
  svm_prob <- predict(svm3,xTest, type="prob")[,2]
  
}


