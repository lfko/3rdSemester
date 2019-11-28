## 11/13/2019
## ML II Workshop 6
## SVM cont'd
set.seed(10111)
library(e1071) # for SVM
library(ggplot2)

# (1) https://www.datacamp.com/community/tutorials/support-vector-machines-r
# First, generate some two-dimensional data
x = matrix(data = rnorm(40), nrow = 20, ncol = 2)
y = rep(c(-1, 1), c(10, 10)) # generate 10 values per class
x[y == 1,] = x[y == 1,] + 1 # for y = 1,moves the means from 0 to 1 in each of the coordinates
plot(x = x, col = y + 3, pch = 19)

# Now process x and y as a data.frame; we need y as a factor
dat = data.frame(x, y = as.factor(y))
svmfit = svm(y ~ ., data = dat, scale = FALSE, kernel = "linear", cost = 10)
print(svmfit) # 6 support vectors, i.e. points that are near to the boundary (or even on the wrong side)
plot(svmfit, data = dat) # marked with a 'X'

# http://uc-r.github.io/svm
# Support Vector Classifier
#x <- matrix(rnorm(20*2), ncol = 2)
x <- matrix(runif(n = 40), ncol = 2) # uniformly distributed
y = c(rep(-1, 10), rep(1, 10))
x[y == 1,] = x[y == 1,] + 1
dat = data.frame(x, y = as.factor(y))
ggplot(data = dat, aes(x = dat$X2, y = dat$X1, color = dat$y, shape = dat$y)) + geom_point(size = 2)
svmfit = svm(y ~ ., data = dat, scale = FALSE, kernel = "linear", cost = 5)
plot(data = dat, svmfit)
