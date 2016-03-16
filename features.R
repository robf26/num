
# features.R

# Finding structure in the data

# Distribution of the factors
data %>%
  melt(id.vars=c("target")) %>% 
  ggplot(aes(variable,value)) + 
  geom_boxplot(aes(fill=as.factor(target))) + 
  coord_flip() 

data %>%
  melt(id.vars=c("target")) %>% 
  ggplot(aes(x=value,fill=variable)) + 
  geom_density(alpha=0.2)

data[trainindex,] %>%
  ggplot(aes(x=feature1,y=feature2,colour=target)) + 
  geom_point(aes(colour=as.factor(target)))
# Structure in the data not being able to take certain values!?
# This will impact tree methods?

# Correlations
corrplot(cor(data[trainindex,feature_names]),"circle",tl.cex=0.6)
corrplot(cor(data[trainindex,feature_names]),"circle",order="hclust",tl.cex=0.6)
corrplot(cor(data[trainindex,c(outcome_name,feature_names)]),"circle",tl.cex=0.6)

# Explore taking PCAs. And also PCA of correlated factors.


# cluster analysis: k-means?






