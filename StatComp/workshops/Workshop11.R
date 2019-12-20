## StatComp 19/20
## Workshop 11 (Confidence Intervals and Hypothesis tests 1)
## 12/20/19

## Ex3
Temp = c(36.8,  37.2,   37.5,   37.0,   36.9,   37.4,   37.9,   38.0)

t_mean = mean(Temp); t_sd = sd(Temp)

# H_0: mean body temperature ~ 37 °C; H_1: mean body temperature != 37 °C

test = t.test(Temp, mu = 37, conf.level = .95)
# t = 2.1355 - test statistic
# check for the critical value with df = 7
t_cr = qt(0.975, 7 - 1)
if(test$statistic < t_cr){
    print("We accept the H_0")
}else{
    print("We reject H_0 and accept H_1")
}

# check using the p-value
test$p.value # .0700995, i.e. the smallest alpha to accept the H_0 is 7%, so it is bigger than 5%, so we accept H_0

## Ex4
load(paste(getwd(), "/3rdSemester/StatComp/data/Fuel.Rda", sep = ""))
# cars per country
table(fuel$country) # 79 JPN; 249 US
# obtain mean and sd per group; the higher the value the better the fuel utilitisation is
tapply(fuel$mpg, fuel$country, mean); tapply(fuel$mpg, fuel$country, sd)
# boxplot mpg per country
boxplot(fuel$mpg ~ fuel$country, data = fuel)
# two-test
t.test(mpg ~ country, data = fuel, var.equal = TRUE) # since both sds are fairly equal
# very small p-value, which indicates, that both means are quite similar -> we reject the H_0