## StatComp 19/20 
## Workshop 6
## 12/17/19
load(paste(getwd(), "/3rdSemester/StatComp/data/grasshoppers50.Rda", sep = "")
)
plot(grasshoppers50)

lm.obj = lm(chirp ~ temp, data = grasshoppers50); summary(lm.obj)
# quadratic regression
lm.obj2 = lm(chirp ~ temp + I(temp^2), data = grasshoppers50); summary(lm.obj2)
coefs_lm.obj2 = coef(lm.obj2)
curve(coefs_lm.obj2[1] + coefs_lm.obj2[2]*x + coefs_lm.obj2[3]*(x^2), from = 3, to = 22, add = TRUE, col="8")

# residual plots
par(mfrow = c(1, 2))
plot(lm.obj, which = 1)
plot(lm.obj2, which = 1)