## StatComp 19/20 
## Workshop 2

gotham.data = read.csv(paste(getwd(), "/data/GCU.csv", sep = ""))

# some basic properties of the data set
dim(gotham.data);str(gotham.data);summary(gotham.data)
nrow(gotham.data) # 26 rows
ncol(gotham.data) # 5 cols
class(gotham.data) # data.frame
names(gotham.data) # variables names

# table of absolute frequencies
table(gotham.data$DegreeSubject)
# table of relative frequencies
prop.table.out = prop.table(table(gotham.data$DegreeSubject)); prop.table.out
# round the output to 3 decimal points and show them as percentage
round(prop.table.out, 3) * 100
# use rbind() to show all the results in a matrix
rbind(round(prop.table.out, 3) * 100, table(gotham.data$DegreeSubject))
# or colum-wise
cbind(round(prop.table.out, 3) * 100, table(gotham.data$DegreeSubject))

# Graphics
# Frequency data > Bar Plots
plot(gotham.data$DegreeSubject, main = "25 Gotham City University Students", sub = "Degree Subject", ylab = "Frequency", col=c("lightgreen","lightpink","lightblue","orange"))
# for continous data > Histogram
hist(gotham.data$Income)
# Obtain the four types frequency for the variable “number of siblings”, and present them all in one table.
table(gotham.data$NSiblings)
prop.table(table(gotham.data$NSiblings))
# Obtain a bar chart for the number of siblings
barplot(gotham.data$NSiblings)
