## StatComp 19/20 
## Workshop 5
## 12/17/19

require(carData)
data(TitanicSurvival)

## Ex1

nrow(TitanicSurvival)
# How many survived/died?
table(TitanicSurvival$survived) # no: 809, yes: 500
prop.table(table(TitanicSurvival$survived)) # no: 0.62, yes: 0.38

# contingency table
table(TitanicSurvival$survived, TitanicSurvival$passengerClass)

# Obtain the overall relative frequencies, the column and row relative frequencies for passenger survivaland class
prop.table(table(TitanicSurvival$survived, TitanicSurvival$passengerClass)) # rel. frequencies
prop.table(table(TitanicSurvival$survived, TitanicSurvival$passengerClass), 1) # rowwise rel. frequencies
prop.table(table(TitanicSurvival$survived, TitanicSurvival$passengerClass), 2) # colwise rel. frequencies

## Ex2
# barchart of passenger survival
barplot(table(TitanicSurvival$survived))
# barchart of passenger survival + class
barplot(table(TitanicSurvival$passengerClass, TitanicSurvival$survived), beside = TRUE, legend = TRUE, col=c("#009999", "#0000FF"))

## Ex4
# boxplot for education split by type
data(Prestige)
boxplot(Prestige$education ~ Prestige$type, col = "lightblue",pch=16)
table(Prestige$type)
# obtain statistics for education split by type
tapply(Prestige$education, Prestige$type, median)
#   bc  prof    wc 
# 8.35 14.44 11.13
mean(Prestige$education, trim = .05) # fraction of observations to be trimmed from each end

# Find the mean and standard deviation forprestigesplit by type
tapply(Prestige$prestige, Prestige$type, mean)
tapply(Prestige$prestige, Prestige$type, sd)
# Obtain the lower and upper quartiles for educationsplit by type
tapply(Prestige$education, Prestige$type, quantile, probs = c(.25, .75))

## Ex5
plot(Prestige$education, Prestige$prestige)
cov(Prestige$education, Prestige$prestige)
cor(Prestige$education, Prestige$prestige) # highly correlated

plot(Prestige$education, Prestige$income)
cov(Prestige$education, Prestige$income)
cor(Prestige$education, Prestige$income) # highly correlated

plot(ecdf(Prestige$income), do.points = FALSE) # CDF