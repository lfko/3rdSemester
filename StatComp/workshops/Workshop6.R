## StatComp 19/20 
## Workshop 6
## 12/17/19

# Ex1
n = 5
occ = c(20, 50, 70, 100, 100) # x
wc = c(25, 35, 20, 30, 45) # y
data = data.frame(occ, wc)

plot(data, col = "red", pch = 5, xlab = "Occupancy", ylab = "Water consumption")
x_mean = mean(occ); y_mean = mean(wc)
var_x = var(occ); s_xy = cov(occ, wc)
sum1 = sum_var(occ, x_mean)
sum2 = sum_var2(occ, wc, x_mean, y_mean)
gradient = s_xy/var_x
intercept = y_mean - gradient * x_mean

sum_var <- function(x, x_mean) {
    sum = 0
    for(j in 1:5){
        sum = (x[j] - x_mean)^2
    }

    return(sum)
}

sum_var2 <- function(x, y, x_mean, y_mean){
    sum = 0
    for(j in 1:5){
        sum = (x[j] - x_mean)*(y[j] - y_mean)
    }

    return(sum)
}

linReg <- function(x, gradient, intercept){
    return(intercept + gradient * x)
}
abline(c(intercept, gradient))
# predict for occ = 70
y_pred = linReg(70, gradient, intercept); y_pred # 31.26068

# Ex2
lm.obj = lm(wc ~ occ, data = data)
summary(lm.obj); fitted(lm.obj)
predict(lm.obj, newdat = data.frame(occ = c(70))) # 31.26068