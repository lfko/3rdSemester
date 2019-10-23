## 10/23/2019
## ML II Workshop 3
## Non-Linear Modeling II (LOESS & GAM)
## @NB: https://m-clark.github.io/generalized-additive-models/preface.html
## @NB: http://r-statistics.co/Loess-Regression-With-R.html
## @NB: http://environmentalcomputing.net/intro-to-gams/
startMar <- par()$mar

# (1) Local Regression
library(MASS); library(splines)
plot(mcycle)
fit1 = smooth.spline(mcycle$times, mcycle$accel, df = 10) # Smooth Splines, 10 Degrees of Freedom
x.grid <- 0:56
preds = predict(fit1, x.grid) 
lines(x.grid, preds$y, lwd = 2, col = "red") # plot the fitted values

# now Loess in one step
loess.fit = loess(lmx0, degree = 1)
loess.fit$fitted
preds = predict(loess.fit, x.grid)
lines(x.grid, preds, lwd = 2 , col = "green")

# (2) GAMs
# James et al. 7.8.3
require(gam)
library(ISLR); 
attach(Wage)

# ramshackled GAM
gam1 = lm(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage); coef(gam1)
gam.m3 = gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage); coef(gam.m3)
par(mfrow = c(1,3))
plot(gam.m3, se = TRUE, col = "blue")
# or ...
plot.Gam(gam.m3, se = TRUE, col = "blue")
