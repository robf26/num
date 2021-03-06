
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
  
  # fitmxnet
  model <- lapply(sampleindex,function(x) fitmxnet(x))
  models <- c(models,list(model))
  print("Finished fitting mxnet")
  
  # Fit other models based on cluster pairs
#  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust1",clust1))
#  models <- c(models,list(model))
#  print("Finished fitting glm_clust1")
  
#  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust2",clust2))
#  models <- c(models,list(model))
#  print("Finished fitting glm_clust2")
  
#  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust3",clust3))
#  models <- c(models,list(model))
#  print("Finished fitting glm_clust3")
  
  model <- lapply(sampleindex,function(x) fitglmsub(x,"glm_clust4",clust4))
  models <- c(models,list(model))
  print("Finished fitting glm_clust4")
  
  # glm net, elasto net
  #model <- lapply(sampleindex,function(x) fitglmnet(x))
  #models <- c(models,list(model))
  #print("Finished fitting glmnet")
  
  # fit nnet
#  model <- lapply(sampleindex,function(x) fitnnet(x))
#  models <- c(models,list(model))
#  print("Finished fitting nnet")
  
  # fit pca glm. Several pcas
  #pca_range = c(3,7)
  #for (q in pca_range) {
  #  model <- lapply(sampleindex,function(x) fitglmpca(x, pcas = q))
  #  models <- c(models,list(model))
  #  print(paste0("Finished fitting glm_pca_",q))
  #}
  
  #model <- lapply(sampleindex,function(x) fitglmpcagroup(x))
  #models <- c(models,list(model))
  #print("Finished fitting glmpca_group1")
  
  # bagged
  model <- lapply(sampleindex,function(x) fitbaggedglm(x))
  models <- c(models,list(model))
  print("Finished fitting baggedglm")
  
  # fitxgboost
  model <- lapply(sampleindex,function(x) fitxgboost(x))
  models <- c(models,list(model))
  print("Finished fitting xgboost")
  
  #model <- lapply(sampleindex,function(x) fitglmpcagroup(x,cols=8:14))
  #models <- c(models,list(model))
  #print("Finished fitting glmpca_group2")
  
  #model <- lapply(sampleindex,function(x) fitglmpcagroup(x,cols=15:21))
  #models <- c(models,list(model))
  #print("Finished fitting glmpca_group3")
  
  return(models)
}

## glm with ability to fit on certain features only ##
fitglmsub <- function(index,
                      name="glm",features="all") {

  samples <- samplingcv("option1",trainindex,testindex,ytrain)
  sampleindex <- lapply(samples,function(x) x$index)
  index = sampleindex[[1]]
  features = clust4
  
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
  submission_glm4 <- data.frame(t_id = testid,
                                 probability = prob$test)
  
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
  # FIX: What should lamda be?
  
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

# Fit a basic neural network
fitnnet <- function(index,
                      name="nnet") {
  samples <- samplingcv("option2",trainindex,testindex,ytrain)
  sampleindex <- lapply(samples,function(x) x$index)
  index = sampleindex[[1]]
  
  model <- list()
  model$name = name
  require(nnet)
  yTrain1 <- data.frame(all)[,outcome_name]
  yNet1 <- class.ind(yTrain1)
  model_fit <- nnet(data[index$train,clust4],yNet1[index$train,],
                size=9,maxit = 500,softmax=TRUE)

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

fitglmpca <- function(index,
                      name="glm_pca_",pcas = 3) {
  
  model <- list()
  model$name = paste0(name,pcas)
  model_fit <- glm(target~ .,
                   data=data_pca_all_scale_c[index$train,c(1:pcas,ncol(data))],
                   family=binomial(logit))
  prob <- lapply(index,function(x) {predict(model_fit,
                                            newdata=data_pca_all_scale_c[x,1:pcas],
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

fitglmpcagroup <- function(index,
                      name="glm_pca_group_",cols = 1:7) {
  
  model <- list()
  model$name = paste(name,cols[1],"_",cols[length(cols)],sep="")
  model_fit <- glm(target~ .,
                   data=data_pca_group[index$train,c(cols,ncol(data))],
                   family=binomial(logit))
  prob <- lapply(index,function(x) {predict(model_fit,
                                            newdata=data_pca_group[x,cols],
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

## Model that runs k-means then runs pca then fits glmnet.


# Fit a deep neural network?
fitmxnet <- function(index,
                    name="mxnet") {
  # To run separately from here...
  require(mxnet)
  #samples <- samplingcv("option2",trainindex,testindex,ytrain)
  #sampleindex <- lapply(samples,function(x) x$index)
  #index = sampleindex[[1]]
  test.x = data.matrix(data[index$test,1:21])
  test.y = data[index$test,outcome_name]
  
  demo.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
    res <- logLoss(label,t(pred[2,]))
    # label is vector, size batch size, n. pred is n by 2 outputs
    return(res)
  })
  
  
  model <- list()
  model$name = "mxnet"
  #require(mxnet)
  
  # Include L2 regularisation wd, to not overfit
  # Dropout to avoid overfitting, but only on 1st layer for convergence?
  data1 <- mx.symbol.Variable("data")
  l1 <- mx.symbol.FullyConnected(data1, num_hidden=21, name="layer1")
  l1 <- mx.symbol.Activation(l1, act_type = "tanh")
  l1 <- mx.symbol.Dropout(l1, p = 0.2)  # Option to include dropout
  l2 <- mx.symbol.FullyConnected(l1, num_hidden=21, name="layer2")
  l2 <- mx.symbol.Activation(l2, act_type = "tanh")
  l2 <- mx.symbol.Dropout(l2, p = 0.5) 
  l3 <- mx.symbol.FullyConnected(l2, num_hidden=21, name="layer3")
  l3 <- mx.symbol.Activation(l3, act_type = "tanh")
  net <- mx.symbol.FullyConnected(l3, num_hidden = 2)
  net <- mx.symbol.SoftmaxOutput(net)

  mx.set.seed(1)
  # 2 layer. wd: 0.00001, learn 0.005, batch 25

  model_fit <- mx.model.FeedForward.create(net, 
                                       X=data.matrix(data[index$train,1:21]), 
                                       y=data[index$train,outcome_name],
                                       ctx=mx.cpu(), num.round=50, 
                                       array.batch.size=50,
                                       momentum=0.9, 
                                       wd= 0.0001,
                                       learning.rate= 0.1,
                                       lr_scheduler=FactorScheduler(1000,0.85,TRUE),
                                       initializer=mx.init.uniform(0.07),
                                       eval.metric=demo.metric.logloss,
                                       eval.data=list(data=test.x, label=test.y))
                                       #eval.metric=mx.metric.accuracy)

  prob <- lapply(index,function(x) 
  {predict(model_fit,data.matrix(data[x,1:21]))[2,]})
  
  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index, prob, SIMPLIFY = FALSE)
  
  #logloss
  #submission_mxnet <- data.frame(t_id = testid,
  #                         probability = prob$test)
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  return(model)
}


# How to stop rf overfitting?
fitrf <- function(index,
                  name="rf") {
  samples <- samplingcv("option2",trainindex,testindex,ytrain)
  sampleindex <- lapply(samples,function(x) x$index)
  index = sampleindex[[1]]
  model <- list()
  model$name = "rf"
  require(randomForest)
  # Need to make response a factor
  data_f <- data_pca_all_scale_c[,c(1:10,22)]
  data_f$target <- factor(data_f$target,labels = c("x0","x1"))
  
  set.seed(1)
  model_fit <- randomForest(target~ .,
                   data=data_f[index$train,],
                   mtry = 5,
                   strata = target,
                   nodesize = 10,
                   ntree = 500)
  
  prob <- lapply(index,function(x) {predict(model_fit,
                                            newdata=data_f[x,],
                                            type="prob")[,2]})
  summary(prob[[1]])
  logloss <- mapply(function(x,y) logLoss(data[x,"target"], y), 
                    index, prob, SIMPLIFY = FALSE)
  logloss
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  #models <- append(models,list(model))
  return(model)
}


fitbaggedglm <- function(index,
                  name="bagged_glm") {
  
  #samples <- samplingcv("option1",trainindex,testindex,ytrain)
  #sampleindex <- lapply(samples,function(x) x$index)
  #index = sampleindex[[1]]
  
  model <- list()
  model$name = "bagged_glm"

  # Need to make response a factor
  data_f <- data_pca_all_scale_c[,c(1:10,22)]
  data_f$target <- factor(data_f$target,labels = c("x0","x1"))
  
  library(foreach)
  set.seed(1)
  training2      <- data_f[index$train,]
  length_divisor <- 2
  iterations     <- 100
  features <- 5
  
  library(doParallel)
  #ptm <- proc.time()
  cl <- makeCluster(detectCores()-1) 
  registerDoParallel(cl)
  bagged <- foreach(m=1:iterations) %dopar% {
    # ,.combine=rbind
    training_positions <- sample(nrow(training2), 
                                 size=floor((nrow(training2)/length_divisor)))
    glm_fit <- glm(target~ . ,
                   data=data_f[training_positions,],
                   family=binomial(logit))
    
    return(list(
      n=m
      ,test.pred=predict(glm_fit,
                         newdata=data_f[index$test,], 
                         type="response")
    ))
  }
  stopCluster(cl)  
  #proc.time() - ptm
  
  ##Extract the pieces
  res.test.pred <- sapply(bagged,function(x) x$test.pred)
  
  #submission_bag <- data.frame(t_id = testid,
  #                             probability = res.test.pred)
  
  #predictions
  prob <- list()
  prob$test <- apply(res.test.pred,1,mean)
  logloss <- logLoss(data[index$test,"target"],prob$test)
  #logloss
  #summary(prob[[1]])
  
  # Save output
  if (saveprob) {
    model$prob <- prob
  }
  model$logloss <- logloss
  #models <- append(models,list(model))
  return(model)
}


fitxgboost <- function(index,
                           name="xgboost") {
# Fit a gbm with xgboost
## Xgboost ##
# Try fitting xgboost as per:
# https://www.kaggle.com/tqchen/otto-group-product-classification-challenge/understanding-xgboost-model-on-otto-data/notebook
library(xgboost)
require(methods)

#samples <- samplingcv("option1",trainindex,testindex,ytrain)
#sampleindex <- lapply(samples,function(x) x$index)
#index = sampleindex[[1]]

model <- list()
model$name = "xgboost"

y <- data_pca_all_scale_c$target[index$train]
x <- as.matrix(data_pca_all_scale_c[index$train,1:21])

# http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees
# nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
xgb_params_1 = list(
  objective = "binary:logistic",  
  eval_metric = "logloss",
  verbose = 1,
  max.depth = 2,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 0.35,
  subsample = 0.25,
  max_delta_step = 5,
  min_child_weight = 20
)

# cross-validate xgboost to get the accurate measure of error
#xgb_cv_1 = xgb.cv(params = xgb_params_1,
#                  data = x,
#                  label = y,
#                  nrounds = 2000, 
#                  nfold = 5,             # number of folds in K-fold
#                  prediction = TRUE,      # return the prediction using the final model 
#                  showsd = TRUE,          # standard deviation of loss across folds
#                  stratified = TRUE,      # sample is unbalanced; use stratified sampling
#                  verbose = TRUE,
#                  print.every.n = 20, 
#                  early.stop.round = 200
#)

# plot the cv metric for the training and testing samples
#xgb_cv_1$dt %>%
#  select(-contains("std")) %>%
#  mutate(IterationNum = 1:n()) %>%
#  gather(TestOrTrain, LogLoss, -IterationNum) %>%
#  ggplot(aes(x = IterationNum, y = LogLoss, group = TestOrTrain, color = TestOrTrain)) + 
#  geom_line() + 
#  theme_bw()

# fit the model with the parameters specified above
xgb_1 = xgboost(data = x,
                label = y,
                params = xgb_params_1,
                nrounds = 500,              # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 20
)

# Compare actual test set prediction error
pred = predict(xgb_1,as.matrix(data_pca_all_scale_c[index$test,1:21])) # prob = TRUE not working?

prob <- list()
prob$test <- pred
logloss <- logLoss(data[index$test,"target"],prob$test)


# Save output
if (saveprob) {
  model$prob <- prob
}
model$logloss <- logloss
#models <- append(models,list(model))
return(model)
}


#submission_xgb <- data.frame(t_id = testid,
#                         probability = pred)
#probs <- data.frame(xgb = submission_xgb[,2], 
#                    mxnet = submission_mxnet[,2], 
#                    bag = submission_bag[,2],
#                    glm4 = submission_glm4[,2])
#cor(probs)
#submission_ens <- data.frame(t_id = testid,
#                             probability = apply(probs,1,mean))
#write_csv(submission_ens,"./predictions/predictions_ens.csv")




