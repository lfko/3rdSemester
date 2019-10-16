## 10/16/2019
## ML II Workshop 2
## Non-Linear Modeling
## @NB: https://datascienceplus.com/cubic-and-smoothing-splines-in-r/
## @NB: https://towardsdatascience.com/unraveling-spline-regression-in-r-937626bc3d96
startMar <- par()$mar

# James et al. Lab 7.8
library(ISLR)
attach(Wage)
dim(Wage); str(Wage); names(Wage)
plot(logwage, education)
hist(logwage); barplot(table(Wage$year))
fit = lm(wage ~ poly(age, 4, raw = TRUE), data = Wage) # fits a linear combination of higher-order variables
coef(summary(fit))

# now prediction for new values of age
# obtain the lower and the upper boundary for the age variable
agelimes = range(age) # 18 80
age.grid = seq(from = agelimes[1], to = agelimes[2])
preds = predict(fit, newdata = data.frame(age = age.grid), se = TRUE) # show also standard errors
se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit-2*preds$se.fit) # SE bands for our predicted values

# plot data and fit
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0)) # define some proper margins
plot(age, wage, xlim = agelimes, cex = 0.5, col = "darkgrey")
title("Degree -4 Polynomial", outer = T)
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "red", lty = 3) # plot additional SE bands; lty = linetype

# Splines
library(splines) # produces, per default, cubic splines
spline.fit = lm(wage ~ bs(age, knots = c(25, 40, 60)), data = Wage)
pred = predict(spline.fit, newdata = data.frame(age = age.grid), se = T)
plot(age, wage, col = "grey")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit + 2*pred$se, lty = "dashed")
lines(age.grid, pred$fit - 2*pred$se, lty = "dashed")

# natural spline
ns.fit = lm(wage ~ ns(age, df = 4), data = Wage)
pred2 = predict(ns.fit, newdata = data.frame(age = age.grid), se = T)
lines(age.grid, pred2$fit, col = "red", lwd = 2)

# smooth spline
plot(age, wage, xlim = agelimes, cex = .5, col = "darkgrey")
title("Smoothing Splines")
sm.fit = smooth.spline(age, wage, df = 16) # using the df as a target parameter for the best lambda (to achieve this df)
sm.fit.cv = smooth.spline(age, wage, cv = TRUE) # using CV to calculate the best lambda aka smoothing penaliser
sm.fit.cv$df # ~ 6.8
lines(sm.fit, col = "red", lwd = 2)
lines(sm.fit.cv, col = "green", lwd = 2)
detach(Wage)

## Motorcycle Helmet data set
library(MASS)
attach(mcycle)
plot(times, accel, xlab = "ms after impact",  ylab = "G force after the impact", main = "Motorcycle Helmet acceleration")
time.grid = seq(from = min(times), to = max(times))

# Polynomial Regression, DF = 4
poly.4.fit = lm(accel ~ poly(times, 4), data = mcycle)
poly.4.pred = predict(poly.4.fit, newdata = data.frame(times = time.grid), se = T)
se.bands = cbind(poly.4.pred$fit + 2 * poly.4.pred$se.fit, poly.4.pred$fit - 2 * poly.4.pred$se.fit)
lines(time.grid, poly.4.pred$fit, lwd = 2, col = "green")
matlines(time.grid, se.bands, lwd = 2, col = "green")

# , DF = 16
poly.16.fit = lm(accel ~ poly(times, 16), data = mcycle)
poly.16.pred = predict(poly.16.fit, newdata = data.frame(times = time.grid), se = T)
se.bands = cbind(poly.16.pred$fit + 2 * poly.16.pred$se.fit, poly.16.pred$fit - 2 * poly.16.pred$se.fit)
lines(time.grid, poly.16.pred$fit, lwd = 2, col = "red")
matlines(time.grid, se.bands, lwd = 2, col = "red")


detach(mcycle)