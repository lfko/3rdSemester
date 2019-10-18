## StatComp 19/20 
## Workshop 3
## 10/18/19

library(car)
data(Prestige) # Canadian Prestige data set

# How many rows and columns? 120, 6
nrow(Prestige); ncol(Prestige); dim(Prestige) 
names(Prestige); str(Prestige)

# Statistical properties of variable income
mean(Prestige$income); median(Prestige$income);sd(Prestige$income) # They are in the actual unit of the variable
var(Prestige$income);sqrt(var(Prestige$income)) # variance is squared! sqrt(var) -> sd()

# Missing values
x <- c(1:5, NA, 7:15)
is.na(x) # logical vector, i.e. boolean indexing
!is.na(x) # reserves the vector
mean(x, na.rm = TRUE) # works!
mean(x, na.rm = FALSE) # nope!
mean(x[!is.na(x)]) # selects on available indexes

# Quantiles
quantile(Prestige$income, probs = c(0.25, 0.75))
