## StatComp 19/20 
## Workshop 8
## 12/19/19

install.packages('datarium')
require(datarium)
data(marketing)

## Ex1
summary(marketing); nrow(marketing); ncol(marketing); str(marketing); names(marketing)
head(marketing)
pairs(marketing) # scatterplots for all possible combinations

# fit a linear model
lin.mod = lm(marketing$sales ~ marketing$youtube, data = marketing); summary(lin.mod)
# second model with newspaper
lin.mod2 = lm(marketing$sales ~ marketing$youtube + marketing$newspaper, data = marketing); summary(lin.mod2)
# third model with facebook
lin.mod3 = lm(marketing$sales ~ marketing$facebook + marketing$youtube + marketing$newspaper, data = marketing); summary(lin.mod3)
# collinearity between newspaper and facebook
cor(marketing$facebook, marketing$newspaper)

# fit fourth model with YT and FB only
lin.mod4 = lm(marketing$sales ~ marketing$youtube + marketing$facebook, data = marketing)
summary(lin.mod4)
# plot residual plot
plot(lin.mod4, which = 1) # there should be no pattern, but there is

# transforming the sales with log; removing extreme outliers
lin.mod5 = lm(log(marketing$sales) ~ marketing$youtube + marketing$facebook, data = marketing, subset = c(-131, -156))
summary(lin.mod5); plot(lin.mod5, which = 1)

# predict new sales; must use exp() because model was fitted with log(sales)
newdat = data.frame(youtube = 180, facebook = 27)
sales.pred = predict(lin.mod5, newdat)

## Ex2
install.packages('ISLR'); require(ISLR)
dim(Auto); names(Auto); str(Auto);
head(Auto)
# add a new variable litres per 100 km to the frame
Auto$fuel = 235/Auto$mpg; Auto$fuel
# linear model I
lin.mod1 = lm(Auto$fuel ~ Auto$weight, data = Auto); summary(lin.mod1)
lin.mod2 = lm(fuel ~ weight + horsepower + displacement, data = Auto); summary(lin.mod2)
plot(lin.mod2, which = 1)

# final prediction
y_pred = predict(lin.mod2, newdat = data.frame(weight = 2500, horsepower = 120, displacement = 150)); y_pred