## StatComp 19/20
## example R test
## 12/19/19
## https://www.dummies.com/programming/r/

## Preliminary
load(paste(getwd(), "/3rdSemester/StatComp/data/TestOneExample.Rda", sep = ""))
loaded()

## Ex1
# sample size
length(Receipts) # 25
# median
median(Receipts) # 18.05
# mean
mean(Receipts) # 21.2348
# the 0.4-quantile
quantile(Receipts, probs = .4) # 16.24
# histogram of the data
hist(Receipts, col = "grey", breaks = seq(from = 0, to = 50, by = 5), 
main = "Receipts", xlab = "Receipts", ylab = "Count")

## Ex2
# number of rows in datf
nrow(datf) # 15
# variable names
names(datf) # "aaa", "ddd", "ggg"
# frequency table
table(datf$ddd)
# box plot with a box for each level of "ddd"
boxplot(datf$aaa ~ datf$ddd)
# mean value of "aaa" for each level of "ddd"
tapply(datf$aaa, datf$ddd, mean)
# append datf2 to datf2 column-wise
cbind(datf, datf2) # row numbers must be equal

## Ex3
set.seed(100)
rnorm(1) # 0.8343671
mu = 10; var = 25; sd = 5
# P(X<=8)
pnorm(8, mu, sd) # 0.3445783
# P(X_<=8)
pnorm(mean(rnorm(30, mu, sd)), mu, sd) # 0.5130785
# generate a sample 30 random numbers
randsamp = rnorm(30, mu, sd)
length(randsamp[randsamp <= 8]) # 11