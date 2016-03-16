
# sampling.R

# Options for sampling:
# Always uses stratefied sampling
# 1: entire training set, and test set is unseen data, hence no cross validation
#     - used for final model, obviously no sampling
# X: another option for basic sample split, no folds etc. 
# 2: train set is split into train set and test set
#     - initially using full train set. Options for folds and repeats 
# 3: train set is split into train, validation, test
#     - useful for cross validating ensemble models 
# 4: train set is split into same sized train and test sets, 
#     - can set proportional increment by number of steps
#     - max size is half the data
# 5: train set is split into incrementing train set size and fixed proportion test set
#     - might be better than option 4 as can train on more than half the data size. 
#     - option to keep test set fixed, have folds, or random every time.
#     - might be able to check bias/variance by correlations of predictions over time?

# Variables used:
# trainindex  : index for training data in data
# testindex   : index for test data in data
# ytrain      : trainindex factor used for stratefied sampling

samplingcv <- function(type,trainindex,testindex,ytrain) {
  
  samples <- list()
  ytrain_n <- length(ytrain)

  if (type=="option1") {
    ## Option 1:
    sample <- list()
    sample$type = 1
    sample$rep = 1
    sample$fold = 0
    index <- list()
    index$train <- trainindex
    index$test <- testindex
    sample$index <- index
    samples <- append(samples,list(sample))
    
  } else if (type=="option2") {
    ## Option 2:
    folds = 5 
    repeats = 1
    
    id = 1
    set.seed(1)
    for (k in 1:repeats) {
      test <- createFolds(ytrain, k = folds)
      train <- lapply(test,function(x) setdiff(trainindex,x))
      for (j in 1:folds) {
        sample <- list()
        sample$type = 2
        sample$id = id
        sample$rep = k
        sample$fold = j
        sample$index$train <- train[[j]]
        sample$index$test <- test[[j]]
        samples <- append(samples,list(sample))
        id <- id +1
      }
    }
  
  } else if (type=="option3") {
    ## Option 3:
    prop_train <- 0.6
    prop_valid <- 0.2
    prop_test <- 0.2
    repeats = 10
    
    id = 1
    set.seed(1)
    for (k in 1:repeats) {
      p_test <- as.vector(createDataPartition(ytrain,p = prop_test,list=FALSE))
      p_trainvalid <- setdiff(trainindex,p_test)
      p_temp <- createDataPartition(ytrain[p_trainvalid],p = prop_valid/(1-prop_test),list=FALSE) 
      p_valid <- p_trainvalid[p_temp]
      p_train <- setdiff(p_trainvalid,p_valid)
      
      sample <- list()
      sample$type = 3
      sample$id = id
      sample$rep = k
      sample$index$train <- p_train
      sample$index$valid <- p_valid
      sample$index$test <- p_test
      samples <- append(samples,list(sample))
      id <- id +1
    }
    
    
  } else if (type=="option4") {
    ## Option 4: 
    intervals = 5
    repeats = 3
    
    id = 1
    set.seed(1)
    for (k in 1:repeats) {
      for (j in 1:intervals) {
        p_range <- as.vector(createDataPartition(ytrain,p = j*1/intervals,list=FALSE))
        p_temp <- createDataPartition(ytrain[p_range],p = 0.5,list=FALSE) 
        p_train <- p_range[p_temp]
        p_test <- setdiff(p_range,p_train)
        
        sample <- list()
        sample$type = 4
        sample$id = id
        sample$rep = k
        sample$interval = j
        sample$n = length(p_train)
        sample$index$train <- p_train
        sample$index$test <- p_test
        samples <- append(samples,list(sample))
        id <- id +1
      }
    }
    
  } else if (type=="option5") {
    # 5: train set is split into incrementing train set size and fixed proportion test set
    #     - might be better than option 4 as can train on more than half the data size. 
    #     - option to keep test set fixed, have folds, or random every time.
    #     - might be able to check bias/variance by correlations of predictions over time?
    prop_test = 0.2
    fold_test = TRUE  # creates more samples by factor of 1/prop_test
    repeat_test = 1
    intervals = 5
    repeat_train = 3
    
    # Randomise to get test set of a particular size. Keep constant. 
    # Or use folds to get multiple test sets that then stay constant.
    # Then randomise to get different sized samples
    
    fold_n = ifelse(fold_test,floor(1/prop_test),1)
    id <- 1
    set.seed(1)
    for (k in 1:repeat_test) {
      # Create test sets
      if (fold_test) {
        test_k <- createFolds(ytrain, k = fold_n)
      } else {
        test_k <- createDataPartition(ytrain,p = prop_test)
      }
      # Loop over test sets
      for (i in 1:fold_n) {
        p_test <- test_k[[i]]
        # Define sampling range for intervals
        p_range <- setdiff(trainindex,p_test)
        
        # Loop through train iterations
        for (z in 1:repeat_train) {
          # Assumption: when adding more data, we want it to include previous data?
          # Assumption: should add data to train on in a stratefied manner?
          p_train_j <- createFolds(ytrain[p_range],k = intervals)
          
          for (j in 1:intervals) {
            if (j==1) {
              p_train <- p_train_j[[1]]
            }else{
              p_train <- c(p_train,p_train_j[[j]])
            }
            
            # Save sample data
            sample <- list()
            sample$type = 5
            sample$id = id
            sample$rep_test = k
            sample$fold = i
            sample$rep_train = z
            sample$interval = j
            sample$n = length(p_train)
            sample$index$train <- p_train
            sample$index$test <- p_test
            samples <- append(samples,list(sample))
            
            id <- id +1
            
          }
        }
      }
    }
    
  } else {
    print("Doesn't recognise type of sampling")
  }
  
  return(samples)
}


extract_pred_as_samples_models_df <- function(models,pred) {
  # Change ordering of nesting of list:
  # Make list of sample ids, containing dataframes 
  # columns as each of the models predictions
  
  # Takes models and pred: train, test, valid
  preds <- list()
  models_n <- length(models)
  models_names <- sapply(models,function(x) x[[1]]$name)
  samples_n <- length(models[[1]])
  for (s in 1:samples_n) {
    for (m in 1:models_n) {
      col <- models[[m]][[s]]$prob[pred]
      # make dataframe
      if (m==1) {
        df <- data.frame(col)
      } else {
        df <- data.frame(df,col)
      }
    }
    # save dataframe
    names(df) <- models_names
    preds[[s]] <- df
  }
  return(preds)
}



