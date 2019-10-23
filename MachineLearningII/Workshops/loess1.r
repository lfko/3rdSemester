#the definintion of the tricube weight function
K <- function(d,maxd) ifelse(maxd > d, (1 - (abs(d)/maxd)^3)^3, 0) # if maxd <= d return 0
library(MASS)

#define your x variable
x <- mcycle$times
#define your outcome variable variable
y <- mcycle$accel  

#loess parameter
span <- 0.4

doLoess <- function(x0idx, span){
  
  #the x value to estimate f(x) using local regression 
  x0 <- x[x0idx]
  n <- length(x)
  ninwindow <- round(span*n)
  
  #we need to find the distance to the furthest point within the window
  windowdist <- sort(abs(x-x0))[ninwindow]
  
  #calculate the weights using the Kernelfunction above. 
  #If the distance is greater than window distance the weight will be zero
  weight <- K(abs(x - x0), windowdist)
  
  length(weight); length(x); length(y)
  #fit a weighted linear regression 
  lmx0 <- lm(y ~ x, weights = weight)
  summary(lmx0)
  
  prx0 = predict(lmx0, newdata = list(x = x0))
  plot(x, y)
  abline(lmx0)
  points(x0, prx0, col = 2, pch = 16)
}

doLoess(28, 0.5)
doLoess(14, 0.75)
