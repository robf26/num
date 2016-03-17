
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

clust <- paste("feature",c(19,1,8,5,17,20,13,
                           15,16,10,11,21,6,4,
                           14,2,3,18,12,7,9),sep="")

# Explore taking PCAs. 
pca1 <- prcomp(data[,feature_names], center = TRUE, scale = TRUE)
summary(pca1)
data_pca_all <- data.frame(pca1$x)
data_pca_all_scale <- scale(data_pca_all)
# also try to chop at 3 stdev?
data_pca_all_scale_c <- data_pca_all_scale
data_pca_all_scale_c[data_pca_all_scale_c > 3] <- 3
data_pca_all_scale_c[data_pca_all_scale_c < -3] <- -3

# And also PCA of correlated factors.
groups <- split(clust,rep(1:7,each = 3))
pcas <- lapply(groups,function(x) prcomp(data[,x], center = TRUE, scale = TRUE))
lapply(pcas,function(x) summary(x))
# First PCA of each group explains ~83%. Third (last) PCA only explains an additional 5%. 
data_pca_group_fac1 <- data.frame(sapply(pcas,function(y) y$x[,1]))
names(data_pca_group_fac1) <- paste("group",1:7,"pca1",sep="")
data_pca_group_fac2 <- data.frame(sapply(pcas,function(y) y$x[,2]))
names(data_pca_group_fac2) <- paste0("group",1:7,"pca2")
data_pca_group_fac3 <- data.frame(sapply(pcas,function(y) y$x[,3]))
names(data_pca_group_fac3) <- paste0("group",1:7,"pca3")
data_pca_group <- bind_cols(data_pca_group_fac1,data_pca_group_fac2,data_pca_group_fac3)


# Spatial sign
#dfs <- spatialSign(data[,feature_names])

# cluster analysis: k-means?






