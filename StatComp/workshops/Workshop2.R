## StatComp 19/20 
## Workshop 2

gotham.data = read.csv("../data/GCU.csv")

# some basic properties of the data set
dim(gotham.data);str(gotham.data);summary(gotham.data)
nrow(gotham.data) # 26 rows
ncol(gotham.data) # 5 cols
class(gotham.data) # data.frame
names(gotham.data) # variables names

# table of absolute frequencies
table(gotham.data$DegreeSubject)
# table of relative frequencies
prop.table(table(gotham.data$DegreeSubject))
