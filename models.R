
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
  # Or loop through for better memory efficiency?
  
  # FIX this code with mapply?
  #logloss <- list()
  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index$train, prob$train)
  
  #logloss$train <- logLoss(data[index$train,"target"], prob$train)
  #if (names(index)[2]=="valid") {
  #  logloss$valid <- logLoss(data[index$valid,"target"], prob$valid)
  #}
  #logloss$test <- logLoss(data[index$test,"target"], prob$test)
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  #models <- append(models,list(model))
  return(model)
}

## glm bagging with random predictors ##


