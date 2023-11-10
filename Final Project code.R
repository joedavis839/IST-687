#Final Project IST 687- Joseph Davis
#DATA is on the pga tour from 2010-2018, and is provided by kaggle user: Jong
#url : https://www.kaggle.com/datasets/jmpark746/pga-tour-data-2010-2018?resource=download

library(tidyverse)
library(readr)
library(caret)
library(rpart)
library(rpart.plot)
library(kernlab)
library(e1071)
pgaTourData20102018 <- read_csv("~/Desktop/Syracuse/Semester 1/Introduction to Data Science/Final Project/pgaTourData20102018.csv")
PGA <- pgaTourData20102018

#Learning a bit about the data

which.max(PGA$Wins)
PGA[998,]
str(PGA)
View(PGA)

#Begin munging

PGA <- PGA[1:1678,]
nrow(PGA)

PGA$Wins[is.na(PGA$Wins)]=0
View(PGA)
PGA$`Top 10`[is.na(PGA$`Top 10`)]=0

#Begin Creating New Data frames

PGA2018 <- PGA[PGA$Year==2018,]
View(PGA2018)

PGA2017 <- PGA[PGA$Year==2017,]
View(PGA2017)

PGA2016 <- PGA[PGA$Year==2016,]
View(PGA2016)

PGA2015 <- PGA[PGA$Year==2015,]
View(PGA2015)

PGA2014 <- PGA[PGA$Year==2014,]
View(PGA2014)

PGA2013 <- PGA[PGA$Year==2013,]
View(PGA2013)

PGA2012 <- PGA[PGA$Year==2012,]
View(PGA2012)

PGA2011 <- PGA[PGA$Year==2011,]
View(PGA2011)

PGA2010 <- PGA[PGA$Year==2010,]
View(PGA2010)

#Descriptive Statistics
#I want to take a look at how much people made in the year 2018
ggplot(PGA2018, aes(x=(parse_number(Money)))) + geom_histogram(bins=12, color="black", fill="green")+ggtitle("2018 Earnings")+ylab("Count")+xlab("Yearly Earnings ($)")
       
#lets make a boxplot describing average score
ggplot(PGA2018, aes(y=PGA2018$`Average Score`)) + geom_boxplot() + ggtitle("2018 Average Score Boxplot") + xlab("Average Score")

#Lets make a scatterplot of Money vs. average score
ggplot(PGA2018, aes(x=(parse_number(PGA2018$Money)), y=PGA2018$`Average Score`)) + geom_point() + ylab("Average Score") + xlab("2018 Earnings") + ggtitle("2018 Average Score vs. Earnings")

#Lets make a scatterplot of Wins finishes vs. average score
ggplot(PGA2018, aes(x=`Wins`, y=`Average Score`))+ geom_point() + ylab("Average Score") + xlab("2018 Wins") + ggtitle("2018 Average Score vs. Wins")

#Lets make a scatterplot of Top10 finishes vs. average score
ggplot(PGA2018, aes(x=`Top 10`, y=`Average Score`))+ geom_point() + ylab("Average Score") + xlab("2018 Top 10 Finishes") + ggtitle("2018 Average Score vs. Top 10 Finishes")

#Making a new dataframe containing the best 25 players from each season

Best2018 <- PGA2018[order(PGA2018$`Average Score`),]
Top25From2018 <- Best2018[1:25,]
Best2017 <- PGA2017[order(PGA2017$`Average Score`),]
Top25From2017 <- Best2017[1:25,]
Best2016 <- PGA2016[order(PGA2016$`Average Score`),]
Top25From2016 <- Best2016[1:25,]
Best2015 <- PGA2015[order(PGA2015$`Average Score`),]
Top25From2015 <- Best2015[1:25,]
Best2014 <- PGA2014[order(PGA2014$`Average Score`),]
Top25From2014 <- Best2014[1:25,]
Best2013 <- PGA2013[order(PGA2013$`Average Score`),]
Top25From2013 <- Best2013[1:25,]
Best2012 <- PGA2012[order(PGA2012$`Average Score`),]
Top25From2012 <- Best2012[1:25,]
Best2011 <- PGA2011[order(PGA2011$`Average Score`),]
Top25From2011 <- Best2011[1:25,]
Best2010 <- PGA2010[order(PGA2010$`Average Score`),]
Top25From2010 <- Best2010[1:25,]

View(Top25From2010)
TopAll <- rbind(Top25From2018, Top25From2017, Top25From2016, Top25From2015, Top25From2014, Top25From2013, Top25From2012, Top25From2011, Top25From2010)

View(TopAll)
TopAll <- TopAll[order(TopAll$`Average Score`),]  
head(TopAll, 10)  

#Linear Model Creation
#Simple scatterplot to show the relationship for the top players
g <- ggplot(TopAll, aes(y=`Fairway Percentage`, x=`Avg Distance`)) +geom_point(aes(color=`Average Score`))
g
g + geom_smooth(method = "lm") + ggtitle("Driving Accuracy vs. Distance") + coord_fixed()


#Linear Model For Driving Accuracy From the 2018 Season
Driving.lm = lm(formula=`Fairway Percentage` ~ `Avg Distance`+`SG:OTT`, data=PGA2018)
summary(Driving.lm)

#How about a linear model that describes average strokes
Strokes.lm = lm(formula=`Average Score` ~ `gir`+`Average Putts`+`Average Scrambling`
                 +`Avg Distance`+`Fairway Percentage`, data=PGA)
summary(Strokes.lm)

#SVM and Recursive Partition Tree

#Create a dataframe with all players and years but an additional column
mean(PGA$`Average Score`)

#New column:If they shot on average below a 70
PGAsub70 <- PGA[PGA$`Average Score`<70,]
View(PGAsub70)  
nrow(PGAsub70)  

#143 rows and our original PGA is 1678, so about the top 8.5%% of players
nrow(PGA)
binaryPGAsub70 <- replace(PGAsub70,"Average Score", 1)
View(binaryPGAsub70)
PGAover70 <- PGA[PGA$`Average Score`>69.9999,]
View(PGAover70)
binaryPGAover70 <- replace(PGAover70,"Average Score", 0)
View(binaryPGAover70)
BinaryPGA <-rbind(binaryPGAsub70, binaryPGAover70)
BinaryPGA$`Average Score` <- as.factor(BinaryPGA$`Average Score`)
View(BinaryPGA)
str(BinaryPGA)
BinaryPGA <- BinaryPGA[, -18]
BinaryPGA <- BinaryPGA[, -1]

#This new BinaryPGA data can be used to make an SVM and rpart

#SVM Model creation with 75% partition of the data
trainList <- createDataPartition(y=BinaryPGA$'Average Score', p=.75, list=FALSE)
testing <- BinaryPGA[-trainList,]
training <- BinaryPGA[trainList,]

#Training the model
fit.svm <- train(`Average Score`~`gir`+`Average Putts`+`Average Scrambling`
                 +`Average SG Putts`+`SG:OTT`+`SG:APR`+`SG:ARG`
                 , data=testing, method="svmRadial",preProc=c("center","scale"))
#Testing the model
predOut <- predict(fit.svm, newdata=testing)
confusion <- confusionMatrix(predOut, testing$`Average Score`)
confusion

fit.svm

#Recursive Partition Model
#Lets make a decision tree
rpart.plot(rpart(BinaryPGA$'Average Score'~BinaryPGA$Rounds+BinaryPGA$'Fairway Percentage'+BinaryPGA$`Avg Distance`
                 +BinaryPGA$gir+BinaryPGA$`Average Putts`+BinaryPGA$`Average Scrambling`+BinaryPGA$Wins
                 +BinaryPGA$`Top 10`+BinaryPGA$`Average SG Putts`+BinaryPGA$`SG:OTT`+BinaryPGA$`SG:APR`
                 +BinaryPGA$`SG:ARG`, data=BinaryPGA))

cartTree <- rpart(BinaryPGA$'Average Score'~BinaryPGA$Rounds+BinaryPGA$'Fairway Percentage'+BinaryPGA$`Avg Distance`
                  +BinaryPGA$gir+BinaryPGA$`Average Putts`+BinaryPGA$`Average Scrambling`+BinaryPGA$Wins
                  +BinaryPGA$`Top 10`+BinaryPGA$`Average SG Putts`+BinaryPGA$`SG:OTT`+BinaryPGA$`SG:APR`
                  +BinaryPGA$`SG:ARG`, data=BinaryPGA)
cartTree
t <- varImp(cartTree)
t%>% arrange(desc(Overall))%>% slice(1:6) 

#I'm curious if we take out wins/top10 how it looks
#####This is our winner#####
rpart.plot(rpart(BinaryPGA$'Average Score'~BinaryPGA$Rounds+BinaryPGA$'Fairway Percentage'+BinaryPGA$`Avg Distance`
                 +BinaryPGA$gir+BinaryPGA$`Average Putts`+BinaryPGA$`Average Scrambling`
                 +BinaryPGA$`Average SG Putts`+BinaryPGA$`SG:OTT`+BinaryPGA$`SG:APR`
                 +BinaryPGA$`SG:ARG`, data=BinaryPGA))

cartTree <- rpart(BinaryPGA$'Average Score'~BinaryPGA$Rounds+BinaryPGA$'Fairway Percentage'+BinaryPGA$`Avg Distance`
                  +BinaryPGA$gir+BinaryPGA$`Average Putts`+BinaryPGA$`Average Scrambling`
                  +BinaryPGA$`Average SG Putts`+BinaryPGA$`SG:OTT`+BinaryPGA$`SG:APR`
                  +BinaryPGA$`SG:ARG`, data=BinaryPGA)
cartTree
t <- varImp(cartTree)
t%>% arrange(desc(Overall))%>% slice(1:6) 
######This is our winner######


#I want a decision tree with just the top 6 biggest predictors
#This made no difference
rpart.plot(rpart(BinaryPGA$`Average Score`~BinaryPGA$`SG:OTT`+BinaryPGA$`Average Scrambling`
                 +BinaryPGA$`SG:APR`+BinaryPGA$`SG:ARG`+BinaryPGA$`Average SG Putts`+BinaryPGA$`Average Putts`, data=BinaryPGA))

cartTree <- rpart(BinaryPGA$`Average Score`~BinaryPGA$`SG:OTT`+BinaryPGA$`Average Scrambling`
                   +BinaryPGA$`SG:APR`+BinaryPGA$`SG:ARG`+BinaryPGA$`Average SG Putts`+BinaryPGA$`Average Putts`, data=BinaryPGA)


cartTree
varImp(cartTree)

#Finally I want to look at professional improvement over time
#What stats are improving year over year?
Changes2010 <- c(mean((PGA2010$`Fairway Percentage`)),mean((PGA2010$`Avg Distance`)),
                 mean(PGA2010$Year), mean(PGA2010$gir), mean(PGA2010$`Average Putts`),
                 mean(PGA2010$`Average Scrambling`), mean(PGA2010$`Average Score`))

Changes2011 <- c(mean((PGA2011$`Fairway Percentage`)),mean((PGA2011$`Avg Distance`)),
                 mean(PGA2011$Year), mean(PGA2011$gir), mean(PGA2011$`Average Putts`),
                 mean(PGA2011$`Average Scrambling`), mean(PGA2011$`Average Score`))

Changes2012 <- c(mean((PGA2012$`Fairway Percentage`)),mean((PGA2012$`Avg Distance`)),
                 mean(PGA2012$Year), mean(PGA2012$gir), mean(PGA2012$`Average Putts`),
                 mean(PGA2012$`Average Scrambling`), mean(PGA2012$`Average Score`))

Changes2013 <- c(mean((PGA2013$`Fairway Percentage`)),mean((PGA2013$`Avg Distance`)),
                 mean(PGA2013$Year), mean(PGA2013$gir), mean(PGA2013$`Average Putts`),
                 mean(PGA2013$`Average Scrambling`), mean(PGA2013$`Average Score`))

Changes2014 <- c(mean((PGA2014$`Fairway Percentage`)),mean((PGA2014$`Avg Distance`)),
                 mean(PGA2014$Year), mean(PGA2014$gir), mean(PGA2014$`Average Putts`),
                 mean(PGA2014$`Average Scrambling`), mean(PGA2014$`Average Score`))

Changes2015 <- c(mean((PGA2015$`Fairway Percentage`)),mean((PGA2015$`Avg Distance`)),
                 mean(PGA2015$Year), mean(PGA2015$gir), mean(PGA2015$`Average Putts`),
                 mean(PGA2015$`Average Scrambling`), mean(PGA2015$`Average Score`))

Changes2016 <- c(mean((PGA2016$`Fairway Percentage`)),mean((PGA2016$`Avg Distance`)),
                 mean(PGA2016$Year), mean(PGA2016$gir), mean(PGA2016$`Average Putts`),
                 mean(PGA2016$`Average Scrambling`), mean(PGA2016$`Average Score`))

Changes2017 <- c(mean((PGA2017$`Fairway Percentage`)),mean((PGA2017$`Avg Distance`)),
                 mean(PGA2017$Year), mean(PGA2017$gir), mean(PGA2017$`Average Putts`),
                 mean(PGA2017$`Average Scrambling`), mean(PGA2017$`Average Score`))

Changes2018 <- c(mean((PGA2018$`Fairway Percentage`)),mean((PGA2018$`Avg Distance`)),
                 mean(PGA2018$Year), mean(PGA2010$gir), mean(PGA2018$`Average Putts`),
                 mean(PGA2018$`Average Scrambling`), mean(PGA2018$`Average Score`))

Changes <- data.frame(Changes2010, Changes2011, Changes2012, Changes2013,
                     Changes2014, Changes2015, Changes2016, Changes2017, Changes2018)
View(Changes)

#Need to flip columns and rows
changesdf <- data.frame(t(Changes))
changesdf
colnames(changesdf) <- c("Fairway Percentage", "Avg Distance", "Year", "gir",
                         "Average Putts", "Average Scrambling", "Average Score")

#Looking at trends for Year vs. Fairway Percentage, greens in regualtion, and Scrambling
g <- ggplot(changesdf, aes(x=`Year`)) +geom_point(aes(y=`Fairway Percentage`, color="Fairway Percentage"))+geom_point(aes(y=`gir`, color="Green in Regulation"))+geom_point(aes(y=`Average Scrambling`, color="Average Scrambling"))+ylab("Percentage")
g + ggtitle("Player Performance over Time")

#Don't seem to be any interesting trends but could still include this
ggplot(changesdf, aes(x=`Year`)) + geom_point(aes(y=`Average Score`, color="Average Putts")) + ylim(70,72) + ggtitle("Player Scores over Time")

#Curious to look at the top 25 players from each year
Changes2010 <- c(mean((Top25From2010$`Fairway Percentage`)),mean((Top25From2010$`Avg Distance`)),
                 mean(Top25From2010$Year), mean(Top25From2010$gir), mean(Top25From2010$`Average Putts`),
                 mean(Top25From2010$`Average Scrambling`), mean(Top25From2010$`Average Score`))

Changes2011 <- c(mean((Top25From2011$`Fairway Percentage`)),mean((Top25From2011$`Avg Distance`)),
                 mean(Top25From2011$Year), mean(Top25From2011$gir), mean(Top25From2011$`Average Putts`),
                 mean(Top25From2011$`Average Scrambling`), mean(Top25From2011$`Average Score`))

Changes2012 <- c(mean((Top25From2012$`Fairway Percentage`)),mean((Top25From2012$`Avg Distance`)),
                 mean(Top25From2012$Year), mean(Top25From2012$gir), mean(Top25From2012$`Average Putts`),
                 mean(Top25From2012$`Average Scrambling`), mean(Top25From2012$`Average Score`))

Changes2013 <- c(mean((Top25From2013$`Fairway Percentage`)),mean((Top25From2013$`Avg Distance`)),
                 mean(Top25From2013$Year), mean(Top25From2013$gir), mean(Top25From2013$`Average Putts`),
                 mean(Top25From2013$`Average Scrambling`), mean(Top25From2013$`Average Score`))

Changes2014 <- c(mean((Top25From2014$`Fairway Percentage`)),mean((Top25From2014$`Avg Distance`)),
                 mean(Top25From2014$Year), mean(Top25From2014$gir), mean(Top25From2014$`Average Putts`),
                 mean(Top25From2014$`Average Scrambling`), mean(Top25From2014$`Average Score`))

Changes2015 <- c(mean((Top25From2015$`Fairway Percentage`)),mean((Top25From2015$`Avg Distance`)),
                 mean(Top25From2015$Year), mean(Top25From2015$gir), mean(Top25From2015$`Average Putts`),
                 mean(Top25From2015$`Average Scrambling`), mean(Top25From2015$`Average Score`))

Changes2016 <- c(mean((Top25From2016$`Fairway Percentage`)),mean((Top25From2016$`Avg Distance`)),
                 mean(PGA2016$Year), mean(Top25From2016$gir), mean(Top25From2016$`Average Putts`),
                 mean(Top25From2016$`Average Scrambling`), mean(Top25From2016$`Average Score`))

Changes2017 <- c(mean((Top25From2017$`Fairway Percentage`)),mean((Top25From2017$`Avg Distance`)),
                 mean(Top25From2017$Year), mean(Top25From2017$gir), mean(Top25From2017$`Average Putts`),
                 mean(Top25From2017$`Average Scrambling`), mean(Top25From2017$`Average Score`))

Changes2018 <- c(mean((Top25From2018$`Fairway Percentage`)),mean((Top25From2018$`Avg Distance`)),
                 mean(Top25From2018$Year), mean(Top25From2010$gir), mean(Top25From2018$`Average Putts`),
                 mean(Top25From2018$`Average Scrambling`), mean(Top25From2018$`Average Score`))

Changes1 <- data.frame(Changes2010, Changes2011, Changes2012, Changes2013,
                      Changes2014, Changes2015, Changes2016, Changes2017, Changes2018)
changesdf1 <- data.frame(t(Changes1))
changesdf1
colnames(changesdf1) <- c("Fairway Percentage", "Avg Distance", "Year", "gir",
                         "Average Putts", "Average Scrambling", "Average Score")

#Looking at trends for Year vs. Fairway Percentage, greens in regualtion, and Scrambling
g2 <- ggplot(changesdf1, aes(x=`Year`)) +geom_point(aes(y=`Fairway Percentage`, color="Fairway Percentage"))+geom_point(aes(y=`gir`, color="Green in Regulation"))+geom_point(aes(y=`Average Scrambling`, color="Average Scrambling"))+ylab("Percentage")
g2 + ggtitle("Top Players Performance over Time")
ggplot(changesdf1, aes(x=`Year`)) + geom_point(aes(y=`Average Score`, color="Average Putts")) + ylim(68,72) + ggtitle("Top Players Scores over Time")

#Lets compare these few stats, seems to be a pattern
ggplot(changesdf, aes(x=`Year`)) +geom_point(data=changesdf, aes(y=`Fairway Percentage`, color="Fairway Percentage"))+geom_point(data=changesdf1, aes(y=`Fairway Percentage`, color="Top Players Fairway Percentage"))+ggtitle("Fairway Percentages")
ggplot(changesdf, aes(x=`Year`)) +geom_point(data=changesdf,aes(y=`gir`, color="Green in Regulation"))+geom_point(data=changesdf1,aes(y=`gir`, color="Top Players Green in Regulation"))+ggtitle("Green in Regulation")
ggplot(changesdf, aes(x=`Year`)) +geom_point(data=changesdf, aes(y=`Average Scrambling`, color="Average Scrambling"))+geom_point(data=changesdf1,aes(y=`Average Scrambling`, color="Top Players Average Scrambling"))+ggtitle("Scrambling")
ggplot(changesdf, aes(x=`Year`)) +geom_point(data=changesdf, aes(y=`Average Score`, color="Average Score"))+geom_point(data=changesdf1,aes(y=`Average Score`, color="Top Players Average Score"))+ggtitle("Average Score")
#Is the difference in means of stats significant?
#Make a new dataset with the best 100 players
PGAorder <- PGA[order(PGA$`Average Score`),]
PGATop100 <- PGAorder[1:100,]
View(PGATop100)

quantile(PGA$`Fairway Percentage`, probs = c(.05, .95))
mean(PGATop100$`Fairway Percentage`)

quantile(PGA$`Avg Distance`, probs = c(.05, .95))
mean(PGATop100$`Avg Distance`)

quantile(PGA$`gir`, probs = c(.05, .95))
mean(PGATop100$`gir`)

quantile(PGA$`Average Putts`, probs = c(.05, .95))
mean(PGATop100$`Average Putts`)

quantile(PGA$`Average Scrambling`, probs = c(.05, .95))
mean(PGATop100$`Average Scrambling`)

#These 2 were signficantly different
quantile(PGA$`Wins`, probs = c(.05, .95))
mean(PGATop100$`Wins`)

quantile(PGA$`Top 10`, probs = c(.05, .95))
mean(PGATop100$`Top 10`)
#

#Lets try at just 90% CI
quantile(PGA$`Fairway Percentage`, probs = c(.1, .9))
mean(PGATop100$`Fairway Percentage`)

quantile(PGA$`Avg Distance`, probs = c(.1, .9))
mean(PGATop100$`Avg Distance`)

quantile(PGA$`gir`, probs = c(.1, .9))
mean(PGATop100$`gir`)

quantile(PGA$`Average Putts`, probs = c(.1, .9))
mean(PGATop100$`Average Putts`)

quantile(PGA$`Average Scrambling`, probs = c(.1, .9))
mean(PGATop100$`Average Scrambling`)





