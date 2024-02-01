

############ 1. Read Training data ############
ziptrain <- read.table(file="zip.train.csv", sep = ",");
ziptrain27 <- subset(ziptrain, ziptrain[,1]==2 | ziptrain[,1]==7); #limit to 2's and 7's

## some sample Exploratory Data Analysis
dim(ziptrain27); ## 1376 257
sum(ziptrain27[,1] == 2); #2=731, 7=645 
summary(ziptrain27); #summary stats
round(cor(ziptrain27),2); #coorelation 


## To see the letter picture of the 5-th row by changing the row observation to a matrix
rowindex = 5; ## You can try other "rowindex" values to see other rows
ziptrain27[rowindex,1];
Xval = t(matrix(data.matrix(ziptrain27[,-1])[rowindex,],byrow=TRUE,16,16)[16:1,]);
image(Xval,col=gray(0:1),axes=FALSE) ## Also try "col=gray(0:32/32)"
image(Xval,col=gray(0:32/32),axes=FALSE)
#

#################### 2. Build Classification Rules #######################


### linear Regression
mod1 <- lm( V1 ~ . , data= ziptrain27);
pred1.train <- predict.lm(mod1, ziptrain27[,-1]);
y1pred.train <- 2 + 5*(pred1.train >= 4.5); ## depending on the indicator variable whether pred1.train >= 4.5 = (2+7)/2.


## Note that we predict Y1 to 2 and 7
mean( y1pred.train != ziptrain27[,1]); #training error
mean( y1pred.train == ziptrain27[,1]); #training accuracy

### knn Classification

library(class);
kk <- seq(1,15,2); # for 8 different k values 
trainerror = rep(x = 0, times = length(kk))
trainacc = rep(x = 0, times = length(kk))
for (i in 1:length(kk)){
  xnew <- ziptrain27[,-1];
  ypred.train <- knn(ziptrain27[,-1], xnew, ziptrain27[,1], k = kk[i]);
  trainerror[i] <- mean(ypred.train  != ziptrain27[,1]); #training error
  trainacc[i] <- mean(ypred.train  == ziptrain27[,1]); #training accuracy
  
}
plot(kk, trainerror)
#print(trainerror)
plot(kk, trainacc)
#print(trainacc)

######################### Testing Error #######################


### read testing data
ziptest <- read.table(file="zip.test.csv", sep = ",");
ziptest27 <- subset(ziptest, ziptest[,1]==2 | ziptest[,1]==7);
dim(ziptest27) ##345 257
sum(ziptest27[,1] == 7); #2=731, 7=645 


### testing error for regression classification rule
pred1.test <- predict.lm(mod1, ziptest27[,-1]);
y1pred.test <- 2 + 5*(pred1.test >= 4.5); ## depending on the indicator variable whether pred1.train >= 4.5 = (2+7)/2.

## Note that we predict Y1 to 2 and 7
mean( y1pred.test != ziptest27[,1]); #testing error
mean( y1pred.test == ziptest27[,1]); #testing accuracy

## Testing error of KNN, and you can change the k values.
library(class);
kk <- seq(1,15,2); 
testerror = rep(x = 0, times = length(kk));
testacc = rep(x = 0, times = length(kk));
#cverror <- rep(x = 0, times = length(kk));
for (i in 1:length(kk)){
  xnew2 <- ziptest27[,-1];
  ypred.test <- knn(ziptrain27[,-1], xnew2, ziptrain27[,1], k = kk[i]);
  testerror[i] <- mean(ypred.test  != ziptest27[,1]); #testing error
  testacc[i] <- mean(ypred.test  == ziptest27[,1]);#testing accuracy
  
}

plot(kk, testerror)
#print(testerror)
plot(kk,testacc)
#print(testacc)


######################## 4. Cross-Validation #####################################



zip27full = rbind(ziptrain27, ziptest27) ### combine to a full data set
n1 = 1376; # training set sample size
n2= 345; # testing set sample size
n = dim(zip27full)[1]; ## the total sample size
set.seed(7406); ### set the seed for randomization



### Initialize the TE values for all models in all $B=100$ loops
B= 100; ### number of loops
TEALL = NULL; ### Final TE values
for (b in 1:B){
  ### randomly select n1 observations as a new training subset in each loop
  flag <- sort(sample(1:n, n1));
  #print(flag)
  zip27traintemp <- zip27full[flag,]; ## temp training set for CV
  zip27testtemp <- zip27full[-flag,]; ## temp testing set for CV
 
  library(class);
  kk =7
  
  cverror <- NULL; 
  
  for (i in 0:kk){ 
    xnew3 <- zip27testtemp[,-1];
    kk <-2*i+1
    #regression
    mod.fulldata <- lm( V1 ~ . , data= zip27traintemp);
    pred.test.fulldata <- predict.lm(mod.fulldata, zip27testtemp[,-1]);
    y1pred.fulldata <- 2 + 5*(pred.test.fulldata  >= 4.5);
    te0 <- mean( y1pred.fulldata  != zip27testtemp[,1]); #regression testing error
    #print(te0)
    
    
    #knn
    ypred.fulldata <- knn(zip27traintemp[,-1], xnew3, zip27traintemp[,1], k = kk);
    testerror <- mean(ypred.fulldata != zip27testtemp[,1]); #knn testing error
    cverror <- cbind(cverror, testerror); 
    #print(cverror)
  }
  TEALL = rbind( TEALL, cbind(te0,cverror));
  #print(TEALL)
}   

dim(TEALL); ### 100X9 matrices

colnames(TEALL) <- c( "linearRegression","KNN1", "KNN3", "KNN5", "KNN7",
                      "KNN9", "KNN11", "KNN13", "KNN15");
## You can report the sample mean/variances of the testing errors so as to compare these models
apply(TEALL, 2, mean);
apply(TEALL, 2, var);

##end##
