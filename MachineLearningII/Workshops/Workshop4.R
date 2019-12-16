## 10/29/2019
## ML II Workshop 4
## Naive-Bayes Learning
install.packages("e1071")
library(e1071)
data("Titanic"); Titanic
# data set is structured frequential, so for better analysis it should be converted to a data frame
tit_df <- as.data.frame(Titanic)