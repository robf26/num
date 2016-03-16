
# Data Manipulation
require(readr)
require(dplyr)
require(tidyr)
require(Matrix)
require(reshape2)

# Data Visualisation 
require(ggplot2)
require(psych)
require(corrplot)

# Machine Learning
require(caret)
require(Metrics)
require(doParallel)
require(xgboost)

### Data Load and Manipulation ###

train <- read_csv("./data/numerai_training_data.csv")
test <- read_csv("./data/numerai_tournament_data.csv")
train$traintest <- "train"
test$traintest <- "test"

all <- bind_rows(train,test)
outcome_name <- "target"
feature_names <- names(all)[1:21]
trainindex <- which(all$traintest=="train")
testindex <- which(all$traintest=="test")
data <- data.frame(all)[,1:22]
ytrain <- data[trainindex,outcome_name]
ytrain_n <- length(ytrain)

testid <- all[testindex,"t_id"]
rm(list=c("train","test")) # drop train and test

### Data Visualisation ###

# Dstribution of the factors
all %>%
  select(-t_id) %>%
  melt(id.vars=c("target","traintest")) %>% 
  ggplot(aes(variable,value)) + 
  geom_boxplot(aes(fill=as.factor(target))) + 
  coord_flip() 

all %>%
  select(-t_id) %>%
  melt(id.vars=c("target","traintest")) %>% 
  ggplot(aes(x=value,fill=variable)) + 
  geom_density(alpha=0.2)

all %>%
  filter(traintest=="train") %>%
  ggplot(aes(x=feature1,y=feature2,colour=target)) + 
  geom_point(aes(colour=as.factor(target)))
# Structure in the data not being able to take certain values!?
# This will impact tree methods?

# Correlations
corrplot(cor(data[trainindex,feature_names]),"circle",tl.cex=0.6)
corrplot(cor(data[trainindex,feature_names]),"circle",order="hclust",tl.cex=0.6)
corrplot(cor(data[trainindex,c(outcome_name,feature_names)]),"circle",tl.cex=0.6)

### Feature Engineering ###

# Make cluster groups:
clust <- paste("feature",c(19,1,8,5,17,20,13,
                           15,16,10,11,21,6,4,
                           14,2,3,18,12,7,9),sep="")
clust1 <- clust[seq(1,21,3)]
clust2 <- clust[seq(2,21,3)]
clust3 <- clust[seq(3,21,3)]

clust4 <- paste("feature",c(1,17,15,10,14,2,12),sep="")


### Call sampling routine ###
source("sampling.R")
samples <- samplingcv("option5",trainindex,testindex,ytrain)
# Extract sampling info:
sample_info = data.frame(t(sapply(samples, function(x) unlist(x[-length(x)]))))
str(lapply(samples,function(x) x$index$train))
str(lapply(samples,function(x) x$index$test))


### Fit models ###
source("models.R")
ptm <- proc.time()
models <- fitallmodels(data,samples,TRUE)
proc.time() - ptm

# Summarise results of models
samples_n <- nrow(sample_info)
models_n <- length(models)
models_names <- sapply(models,function(x) x[[1]]$name)

# How best to extract info? rows: samples, cols: models
logloss_train <- data.frame(sapply(models,function(x) sapply(x,function(x) x$logloss$train)))
logloss_test <- data.frame(sapply(models,function(x) sapply(x,function(x) x$logloss$test)))
names(logloss_train) <- models_names
names(logloss_test) <- models_names
logloss_test

# Tidy output
logloss.train <- data.frame(logloss_train,id = sample_info$id) %>% 
                    gather(model,logloss.train,-id) 
logloss.test <- data.frame(logloss_test,id = sample_info$id) %>% 
                    gather(model,logloss.test,-id)
metrics <- full_join(logloss.train,logloss.test,by=c("id","model"))

# Gather all data
fit <- sample_info
fit <- full_join(fit,metrics,by="id")

## Charts for showing fit relative to sample size:

# Test
fit %>% 
  group_by(model,interval) %>%
  summarise(ll.test = mean(logloss.test)) %>%
  ggplot(aes(x=interval,y=ll.test,colour=model)) +
  geom_line(size=2)
# Plot together as facet
fit %>% 
  gather(metric,value,-(type:model))  %>% 
  group_by(model,interval,metric) %>%
  summarise(ll = mean(value)) %>%
  ggplot(aes(x=interval,y=ll,colour=model)) +
  geom_line(size=2) + 
  facet_wrap(~metric)

# Or show the distribution of the test set 
fit %>% 
  ggplot(aes(x=as.factor(interval),y=logloss.test)) +
  geom_boxplot(aes(fill=as.factor(model)))


## Ensembling models ##
samples <- samplingcv("option3",trainindex,testindex,ytrain)
ptm <- proc.time()
models <- fitallmodels(data,samples,TRUE)
proc.time() - ptm

# Extract predictions to use for correlation analysis
prob_valid <- extract_pred_as_samples_models_df(models,"valid") 
#prob_test <- extract_pred_as_samples_models_df(models,"test") 

# Average correlation with other models, in validation and test set,
cor_valid_samples <- sapply(prob_valid,function(x) cor(x))
cor_valid <- matrix(rowMeans(cor_valid_samples),models_n)
cor_valid

# Extract the actual y's using indices which may be used for fitting ensemble. 
# And used for testing all other results.
samples_validindex_y <- lapply(lapply(samples, function(x) x$index$valid), 
                               function(y) data[y,outcome_name])
samples_testindex_y <- lapply(lapply(samples, function(x) x$index$test), 
                              function(y) data[y,outcome_name])

## Fitting ensemble models ## 
source("ensemble.R")
models_ens <- fitallensembles(models,samples_validindex_y)

models_all <- c(models,models_ens)

# Recalculate performance metrics.
prob_test_ens <- extract_pred_as_samples_models_df(models_all,"test") 
ens_logloss_test <- mapply(function(y,x) apply(x,2,function(z) logLoss(y,z)), 
                                 samples_testindex_y,prob_test_ens)

# Averaged across all samples
apply(ens_logloss_test,1,mean)

# Make charts
data.frame(t(ens_logloss_test)) %>%
  gather(model,logloss.test) %>% 
  ggplot(aes(x=model,y=logloss.test)) + geom_boxplot()


### Submission file ###
samples <- samplingcv("option1",trainindex,testindex,ytrain)
ptm <- proc.time()
models <- fitallmodels(data,samples,TRUE)
proc.time() - ptm
prob_test_ens <- extract_pred_as_samples_models_df(models,"test") 
prob_test_avg <- ensemble_average_pred(prob_test_ens,"test")

submission <- data.frame(t_id = testid,
                         probability = prob_test_avg[[1]]$prob$test)

write_csv(submission,"./predictions/predictions_1.csv")



