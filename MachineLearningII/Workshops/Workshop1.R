## ML II 
## Workshop I
## 09/10/2019
require(mice)
require(VIM)
require(NHANES)

# save the starting margins for the plotting device
startMar <- par()$mar

# Exercise 2
# https://rpubs.com/sediaz/na_aggr

# (a)
data(tao) # load the Tropical Atmosphere Ocean data set
dim(tao) # 736 x 8
str(tao) # 8 numerical variables; UWind (daily average east-west wind), VWind (north-south)
length(which(is.na(tao))) # check how many NAs are there in total -> 177
length(which(complete.cases(tao))) # how many rows do not contain NAs at all
tao$Year<-as.factor(tao$Year) 
# from 'VIM': visualises the distribution of the NAs
aggr(tao)
# The plot on the left shows the percentage of missing values for each variable
# The plot on the right shows the combination of missing values between variables
summary(aggr(tao)) # or as figures
md.pattern(tao, rotate.names=TRUE) 

# (b)
# There 93 observations, which have missing values for Humidity. 
# There are in total 4 observations, which have more than 1 missing value.

# (c)
# fit a linear multivariate regression model, using only the fully known observations, i.e. those without missing values
# first extract the rows with non-missing variables
tao.miss <- !complete.cases(tao)
tao.non.miss <- tao[complete.cases(tao),]
tao.mod <- lm(Sea.Surface.Temp ~ Year + Latitude + Longitude + UWind + VWind + Air.Temp + Humidity, data = tao.non.miss)
summary(tao.mod)
# (d)
# The problem with univariate imputation algorithms is, that they are establishing any link between the variables,
# i.e. they do not assume any co-dependency at all

# Exercise 3
# (a) mean imputation/replacement
mean.replace <- function(x){
  idx.missing <- which(is.na(x)) # get the indixes of actual missing values
  known.mean <- mean(x, na.rm = T) # compute the mean, remove the missing values
  x[idx.missing] <- known.mean # replace the missing values with the mean
  
  return (x)
}
# test it
x <- c(1:10, NA, NA, 95)
mean.replace(x)

hist(tao$Air.Temp) # before replacement
length(which(is.na(tao$Air.Temp))) # 81 missing values
mrep.Air.Temp<-mean.replace(tao$Air.Temp) #impute the Air.Temp using mean replacement
hist(mrep.Air.Temp) # after replacement
length(mrep.Air.Temp) # now all rows have values

# now for the other variables as well
tao.mrep<-tao
tao.mrep$Air.Temp<-mrep.Air.Temp
tao.mrep$Sea.Surface.Temp<-mean.replace(tao$Sea.Surface.Temp)
tao.mrep$Humidity<-mean.replace(tao$Humidity)
summary(aggr(tao.mrep)) # no more missing values. Yay!

with(tao.mrep,plot(Air.Temp,Sea.Surface.Temp,col=Year))
with(tao.mrep,plot(Air.Temp,Sea.Surface.Temp,col=1+tao.miss)) #colour the imputed vales red. 
lm.mrep <- lm(tao.mod$call, data = tao.mrep)
summary(lm.mrep)

# mean/variance simulation
mean.sd.replace<-function(x) 
{
  idx<-which(is.na(x))
  known.mean<-mean(x,na.rm=T) 
  known.sd<-sd(x, na.rm = T) 
  x[idx] <- rnorm(length(idx), known.mean, known.sd) # compute the missing values assuming a normal distribution
  
  return(x)
}  
mean.sd.replace(x) # test it
hist(tao$Air.Temp) # before replacement
length(which(is.na(tao$Air.Temp))) # 81 missing values
msdrep.Air.Temp <- mean.sd.replace(tao$Air.Temp) #impute the Air.Temp using mean replacement
plot(msdrep.Air.Temp)

tao.msdrep <- tao
tao.msdrep$Air.Temp <- msdrep.Air.Temp
tao.msdrep$Sea.Surface.Temp <- mean.sd.replace(tao$Sea.Surface.Temp)
tao.msdrep$Humidity <- mean.sd.replace(tao$Humidity)
with(tao.msdrep, plot(Air.Temp,Sea.Surface.Temp,col=Year))
with(tao.msdrep, plot(Air.Temp, Sea.Surface.Temp, col=1+tao.miss))
lm.msdrep<-lm(tao.mod$call, data = tao.msdrep)
summary(lm.msdrep)

