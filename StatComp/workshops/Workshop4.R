## StatComp 19/20 
## Workshop 4
## 10/27/19
require(car)
data("Prestige")

# plot data for variable income in a histogram
hist(Prestige$income, breaks = 15) # number of bins set to 15
xbar <- mean(Prestige$income)
devs <- Prestige$income - xbar # deviations
sqdevs <-(devs)^2 # squared deviance
ssqdevs <- sum(sqdevs) # sum of all squared deviances
varincome <- ssqdevs/(length(Prestige$income)); varincome
sdincome <- sqrt(varincome); sdincome
var(Prestige$income); sd(Prestige$income)
