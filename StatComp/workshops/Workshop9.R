## StatComp 19/20
## Workshop 9 (Simulation, Sampling andthe Central Limit Theorem)
## 12/02/19

# draw random numbers from a uniform distribution
xsamp = runif(1000)
# probability distribution of uniform; cdf for uniform
curve(punif(x), 0, 1)
plot(ecdf(xsamp), verticals = TRUE, do.points = FALSE, add = TRUE)

# generate values between a and b
xsamp = runif(10, -5, 20)
curve(punif(x, -5, 20), -5, 20)
plot(ecdf(xsamp), verticals = TRUE, do.points = FALSE, add = TRUE)

# normal distribution
# rnorm - random generator
sd = 5
mean = 100
rnorm(5, mean, sd) # uses the sd, not the variance (which is sd²)
# dnorm - probability density function 
dnorm(97, 100, 5) # 0.067
# pnorm - cumulative density function
x = pnorm(97, 100, 5) # 0.27
# qnorm - quantile function; inverse of pnorm
qnorm(x, 100, 5) # the 0.27... quantile is 97!