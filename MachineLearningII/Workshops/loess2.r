#the definintion of the tricube weight function
K <- function(d, maxd) {
  ifelse(maxd > d, (1 - (abs(d)/maxd)^3)^3, 0)
}
#define your x variable
x <- mcycle$times
#define your outcome variable variable
y <- mcycle$accel 
  
##define x.grid the output coordinates for the loess curve
x.grid <- seq(min(x), max(x), length = 100)  
  
span <- 0.4
n <- length(x)
ninwindow <- round(span*n)
yloess <- rep(0, length(x.grid))
for(i in 1:length(x.grid)){
  x0 <- x.grid[i] 
  windowdist <- sort(abs(x - x0))[ninwindow]
  weight <- K(abs(x - x0), windowdist)

  lmx0<-lm(y ~ x, weights = weight)
  
  yloess[i]<-predict(lmx0, data.frame(x = x0))
  
}

plot(x,y)
lines(x.grid, yloess,col ="blue")
