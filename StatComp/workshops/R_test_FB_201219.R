## 12/20/19
## Florian Becker (885187)
## StatComp R-Test
##

## Preliminaries
load("../data/RTestA.Rda")
loaded() # works, although it shows Session B ;-)

## Ex1
nrow(DataA) # 34
ncol(DataA) # 4
names(DataA) # "distance", "vehicle", "passengers", "age"
round(mean(DataA$distance), 4)
round(sd(DataA$distance), 4)
quantile(DataA$distance, probs = c(.1, .9))
hist(DataA$distance, main = "Distance Distribution", xlab = "Distance (km)")
table(DataA$vehicle)
str(DataA$vehicle) # for checking the dtype
boxplot(DataA$distance ~ DataA$vehicle)
table(DataA$vehicle, DataA$passengers)

## Ex2
moment1 <- function(x, m = 3){
    n = length(x)
    sum = 0
    for(i in 1:n){
      sum = sum + ((x[i] - mean(x))**m)
    }
    sum = (1/n)*sum
    
    return(sum)
}
moment1(DataA$distance, 3) # 364.3176
moment1(DataA$distance, 4) # 7277.992

moment2 <- function(x, m = 3){
  sum = cumsum((DataA$distance - mean(DataA$distance))**m) * (1/length(DataA$distance))
  return(tail(sum, n = 1)) # returns the last index, i.e. the cumulative sum
}
moment2(DataA$distance, 3) # 364.3176
moment2(DataA$distance, 4) # 7277.992

# Finish
CheckValue(885187) # 19
