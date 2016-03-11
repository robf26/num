
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
require(doParallel)
require(xgboost)

require(Metrics)

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
  ggplot(aes(x=feature1,y=feature2)) + 
  geom_point()
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
samples <- samplingcv("option3",trainindex,testindex,ytrain)
# Check how to feed these into caret?
# Extract sampling info:
sample_info = data.frame(t(sapply(samples, function(x) unlist(x[-length(x)]))))
str(lapply(samples,function(x) x$index$train))
str(lapply(samples,function(x) x$index$test))


### Fit models ###
source("models.R")
ptm <- proc.time()
models <- fitallmodels(data,samples,TRUE)
proc.time() - ptm

# Create ensemble models here?


# Summarise results of models
samples_n <- nrow(sample_info)
models_n <- length(models)
models_names <- sapply(models,function(x) x[[1]]$name)

# How best to extract info? rows: samples, cols: models
logloss_train <- data.frame(sapply(models,function(x) sapply(x,function(x) x$logloss$train)))
logloss_test <- data.frame(sapply(models,function(x) sapply(x,function(x) x$logloss$test)))
logloss_valid <- data.frame(sapply(models,function(x) sapply(x,function(x) x$logloss$valid)))

names(logloss_train) <- models_names
names(logloss_test) <- models_names
names(logloss_valid) <- models_names

logloss_test

# Tidy output
logloss.train <- data.frame(logloss_train,id = sample_info$id) %>% 
                    gather(model,logloss.train,-id) 
logloss.test <- data.frame(logloss_test,id = sample_info$id) %>% 
                    gather(model,logloss.test,-id)
logloss.valid <- data.frame(logloss_valid,id = sample_info$id) %>% 
                    gather(model,logloss.valid,-id)

metrics <- full_join(logloss.train,logloss.test,by=c("id","model"))
metrics <- full_join(metrics,logloss.valid,by=c("id","model"))

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

# Train
fit %>% 
  group_by(model,interval) %>%
  summarise(ll.train = mean(logloss.train)) %>%
  ggplot(aes(x=interval,y=ll.train,colour=model)) +
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


## Correlations among models ##
# Extract predictions to use for correlation analysis

# Change ordering of nesting of list:
# Validation probs. List of sample ids, 10 elements, of dataframes with 
# columns as each of the models.
prob_valid <- list()
for (s in 1:samples_n) {
  for (m in 1:models_n) {
    col <- models[[m]][[s]]$prob$valid
    # make dataframe
    if (m==1) {
      df <- data.frame(col)
    } else {
      df <- data.frame(df,col)
    }
  }
  # save dataframe
  names(df) <- models_names
  prob_valid[[s]] <- df
}

# Combine this into a function. Or change the fit approach?
prob_test <- list()
for (s in 1:samples_n) {
  for (m in 1:models_n) {
    col <- models[[m]][[s]]$prob$test
    # make dataframe
    if (m==1) {
      df <- data.frame(col)
    } else {
      df <- data.frame(df,col)
    }
  }
  # save dataframe
  names(df) <- models_names
  prob_test[[s]] <- df
}

# When do we want the data as model_sample or sample_model?!

# Average correlation with other models, in test set say,
test_cor <- sapply(prob_test,function(x) cor(x))
test_cor_avg <- matrix(rowMeans(test_cor),models_n)
test_cor_avg

# Create ensemble models at this step? Separate module of functions?
# Feed in list of samples_models

ensemble_prob_valid <- lapply(prob_valid,function(x) rowMeans(x))
#ensemble_prob_valid <- lapply(prob_valid,function(x) apply(x,1,function(z) mean(z[2:4])))
# Code to create ensembles of equal weights across all models. 
# 
prob_valid[[1]]

ensemble_prob_test <- lapply(prob_test,function(x) rowMeans(x))

# Extract the actual y's using indices
samples_validindex_y <- lapply(lapply(samples, function(x) x$index$valid), 
                               function(y) data[y,outcome_name])
samples_testindex_y <- lapply(lapply(samples, function(x) x$index$test), 
                               function(y) data[y,outcome_name])

# Fit logLoss, by sample
ensemble_logloss_valid <- mapply(function(x,y) logLoss(x,y), samples_validindex_y,ensemble_prob_valid)
ensemble_logloss_test <- mapply(function(x,y) logLoss(x,y), samples_testindex_y,ensemble_prob_test)
# Averaged across all samples
mean(ensemble_logloss_valid)
mean(ensemble_logloss_test)

df_models <- fit[,c("rep","model","logloss.valid")]
df_ens <- data.frame(rep = 1:10,model="ensemble",logloss.valid = ensemble_logloss_valid)
df_all <- rbind(df_models,df_ens)
df_all %>% ggplot(aes(x=rep,y=logloss.valid,colour=model)) + geom_point(size=3)

df_all %>% ggplot(aes(x=model,y=logloss.valid)) + geom_boxplot()

# Code to optimise weights on validation set. 


# Then cross validate on test set.



# Validation logloss calculated from probability output. 
# same as logloss_valid and similar to data in fit
# rows: models, cols: iterations
logloss_valid_2 <- mapply(function(x,y) 
          sapply(1:models_n,function(z) logLoss(y,x[,z])), 
          prob_valid, samples_validindex_y)



### Submission file ###

submission <- data.frame(t_id = all[testindex,"t_id"],
                         probability = ensemble_prob_test)

write_csv(submission,"./predictions/predictions_7.csv")



