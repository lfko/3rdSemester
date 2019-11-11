## 11/06/2019
## ML II Workshop 5
## SVM
set.seed(1)
#require(ISLR); require(MASS); require(e1071); require(gbm)
require(e1071)

# (1) James Lab
# small cost - wide margin area
# high cost - small margin area
# generate data for two classes and see if they are nicely separable
x = matrix(rnorm(20 * 2), ncol = 2)
y = c(rep(-1, 10), rep(1, 10))
x[y==1, ] = x[y==1, ] + 1 # generate two classes, i.e. -1 and 1 respectively
plot(x, col = (3 - y))
# they are not clearly separable
dat = data.frame(x = x, y = as.factor(y)) # SVM needs the response to be a factor
svmfit = svm(y~., data = dat, kernel = "linear", cost = 0.10, scale = FALSE) # do not scale to mean=0 or sd=1
plot(svmfit, dat)

# show the support vectors
svmfit$index
summary(svmfit)
# with smaller cost the margin is wider, thus giving us more support vectors now
# use tune() to perform CV on our data and for SVM for several cost levels
tune.out = tune(svm, y~., data = dat, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out) # cost=0.1 results in the lowest cv error
bestmod = tune.out$best.model
summary(bestmod); plot(bestmod, dat, main = "Best Mod after CV")

# Prediction
xtest = matrix(rnorm(20 * 2), ncol = 2)
ytest = sample(c(-1, 1), 20, rep = TRUE) # test response
xtest[ytest == 1, ] = xtest[ytest == 1, ] + 1
testdat = data.frame(x = xtest, y = as.factor(ytest))

ypred = predict(bestmod, testdat)
table(predict = ypred, truth = testdat$y)

# (2) birth weight data set
require(MASS); data("birthwt")
# obtain a classifier for the variable "low" (indicator of birth weight less than 2.5 kg)
# bwt should be binned then
birthwt$low <- as.factor(birthwt$low) # factor for the response
plot(birthwt)
svmfit = svm(low ~. - bwt, data = birthwt, kernel = "linear", cost = 0.10, scale = FALSE) # do not scale to mean=0 or sd=1
#table(svmfit$fitted, birthwt$low)
pred = predict(svmfit, birthwt)
table(pred, birthwt$low)
plot(svmfit, birthwt)
