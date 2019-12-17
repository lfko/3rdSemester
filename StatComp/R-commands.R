# Absolute frequency: The raw counts for each value.
# Relative frequency: The absolute frequencies divided by the sample sizenoften refered to as proportions.
# Cumulative frequency: If there is a sensibleorder then we can add up the (absolute or relative) frequencies cumulatively.
table() # table of absolute frequencies
prop.table(table()) # relative frequencies
(c/r)bind(prop.table(), table()) # show both relative and absolute frequencies column or row-wise

## Structured Data
nrow() # No. of rows
ncol() # No. of columns
dim() # dimensions
names() # variable names

## Probabilities ##
# The probability of three heads from 5 coin tosses is P(X==5)
dbinom(3,5,0.5)
# The probability of three heads or fewer from 5 coin tosses is P(X<=5)
pbinom(3,5,0.5)

## Variable statistics
mean()
median()
var()
sd()
quantile(x, probs = seq(0, 1, 0.25))