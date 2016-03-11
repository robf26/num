
# models.R
# model functions:
# Take an index list including train, and optional valid, test
# returns the model object with risk metrics, and optional prediction probabilities.

# variables: 
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
  
  write_csv(submission,"./predictions/predictions_8.csv")
  
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
  
  # Fit a basic neural network?
  require(nnet)
  yTrain1 <- data.frame(all)[trainindex1,outcome_name]
  
  yNet1 <- class.ind(yTrain)
  nnet1 <- nnet(xTrain[trainindex1,clust1],yNet1[trainindex1,],
                size=9,maxit = 500,softmax=TRUE)
  # Change the decay?
  nnet1_prob <- predict(nnet1,xTrain[trainindex2,clust1])[,2]
  summary(nnet1_prob)
  logLoss(data.frame(all)[trainindex2,outcome_name], 
          as.numeric(nnet1_prob))
  
  # Try knn
  require(class)
  xTrainExValid <- data.frame(all)[trainindex1,feature_names]
  yTrainExValid <- factor(data.frame(all)[trainindex1,outcome_name],,c("x0","x1"))
  xValid <- data.frame(all)[trainindex2,feature_names]
  yValid <- factor(data.frame(all)[trainindex2,outcome_name],,c("x0","x1"))
  
  knn1 <- knn(xTrainExValid[,clust4], xValid[,clust4], yTrainExValid, 
              k = 50, l = 0, prob = TRUE, use.all = TRUE)
  # Bit tricky to extract probabilities
  knn_pred <- as.numeric(knn1)-1
  
  knn_prob1 <- attr(knn1, "prob")
  knn_prob <- ifelse(knn_pred==1,knn_prob1,1-knn_prob1)
  summary(knn_prob)
  logLoss(actualv,knn_prob)
  # hmm.. not working
  # how does it work with many factors..
  
  # No information
  logLoss(actual,rep(0.5,length(actual)))
  # Random noise?
  
  
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
  
  
  
  
  #######
  # svm bagging
  set.seed(1)
  training2 <- data.frame(target=yTrain,xTrain)
  
  bags <- createFolds(yTrain,5)
  yTrain1 <- data.frame(all)[trainindex,outcome_name]
  
  probs4 <- NA
  fitsvmpredict <- function(index) {
    #index = bags[[4]]
    svm_fit <- ksvm(target~.,
                    data=training2[index,
                                   c("target",clust4)],
                    kernel="rbfdot",C=0.5,
                    prob.model=TRUE)
    #predict(svm_fit, xTest, type="probabilities")[,2]  
    probs <- predict(svm_fit, xTrain[-index,], type="probabilities")[,2] 
    print(logLoss(yTrain1[-index],probs))
    probsTest <- predict(svm_fit, xTest, type="probabilities")[,2]
    probs4 <- cbind(probs4,probsTest)
  }
  
  for (k in 1:5) {
    fitsvmpredict(bags[[k]])
  }
  
  
}


