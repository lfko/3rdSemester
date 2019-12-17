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

# linear transformation
linTransform <- function(x) {
    xmax = max(Prestige$income); xmin = min(Prestige$income)
    y = (x/(xmax - xmin)) - (xmin/(xmax - xmin))

    return(y)
}
# apply transformation
Prestige$incomeTransform = linTransform(Prestige$income); Prestige$incomeTransform
# calculate sd
sd(Prestige$incomeTransform) # 0.1680355
range(Prestige$incomeTransform) # all values are between 0 and 1

# Rough interpretation of SD
# intervals [mean - 2*sd; mean + 2*sd]
lowerBound = mean(Prestige$education) - 2*sd(Prestige$education); upperBound = mean(Prestige$education) + 2*sd(Prestige$education)
# how many values in these boundaries?
vals = length(Prestige$education[lowerBound:upperBound]) # 11
quantile(vals)