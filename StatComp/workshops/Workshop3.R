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
quantile(Prestige$income, probs = c(0.25, 0.75)) # obtains lower and upper quartiles

# mode of type?
table(Prestige$type) # bc: 44
which.max(table(Prestige$type)) # also bc, it shows the position in the table
table(Prestige$type == 'bc')[2] + table(Prestige$type == 'wc')[2] + table(Prestige$type == 'prof')[2]# sum of types = 98, 4 NAs
row.names(Prestige)[is.na(Prestige$type)] # where are the missing values?

## Boxplot
boxplot(prestige ~ type, data = Prestige, xlab = "job type", col = "lightblue",pch=16) # they are ordered 
# change factors ordering
Prestige$newtype = factor(Prestige$type, levels = c("bc", "wc", "prof"))
boxplot(prestige ~ newtype, data = Prestige, xlab = "job type", col = "lightblue",pch=16) # they are ordered 
# strip charts are better if there are few rows
stripchart(prestige ~ newtype, data = Prestige, ylab="job type")
stripchart(prestige ~ newtype, data = Prestige, ylab="job type", method="stack")

## Matrixes
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2) # assigning column-wise -> byrow = TRUE
M[1:2, 1] # 1st and 2nd rows, 1st column
M[2, c(1,3)] # 2nd row, 1st and 3rd columns
M[1:2, c(1,3)] # 1st and 2nd rows, 1st and 3rd columns
M[ ,3] # get one complete column
M[1, ] # one complete row
M[, -3] # drop this column

# A + B > Matrix Addition; 2*A > Scalar Multiplication
# A * B > elementwise; A %*% B > Matrix Multiplication
# solve() > inverse