# Practical-maching-Learning
Practical machine learning
library(caret)
library(dplyr)

##Exploring analysis
table(pml.training$classe)
nrow(pml.training)

##Since my pm.training sample is large enough, I like to split
## some of that to testing as well

inTrain <- createDataPartition(y=pml.training$classe, p=0.75, list=FALSE)
training <- pml.training[inTrain,]
testing <- pml.training[-inTrain,]

names(training)
nsv <- nearZeroVar(training,saveMetrics=TRUE)
names(nsv)
nsv1 <- nsv[nsv$nzv == "TRUE",]
training <- tbl_df(training)
##want to remove variables that are near zero or have a lot of missing values since they will not contribute much to the prediction of classe
myvars <- names(training) %in% c("new_window", "kurtosis_roll_belt","kurtosis_picth_belt","kurtosis_yaw_belt","skewness_roll_belt","skewness_roll_belt.1","skewness_yaw_belt","max_yaw_belt",
"min_yaw_belt","amplitude_yaw_belt","avg_roll_arm","stddev_roll_arm","var_roll_arm","avg_pitch_arm","stddev_pitch_arm","var_pitch_arm","avg_yaw_arm",
"stddev_yaw_arm","var_yaw_arm","kurtosis_roll_arm","kurtosis_picth_arm","kurtosis_yaw_arm","max_roll_arm","min_roll_arm","min_pitch_arm","amplitude_roll_arm",
"kurtosis_roll_dumbbell","kurtosis_picth_dumbbell","kurtosis_yaw_dumbbell","skewness_roll_dumbbell","skewness_pitch_dumbbell","skewness_yaw_dumbbell",
"max_yaw_dumbbell","min_yaw_dumbbell","amplitude_yaw_dumbbell","kurtosis_roll_forearm","kurtosis_picth_forearm","kurtosis_yaw_forearm","skewness_roll_forearm","skewness_pitch_forearm",
"skewness_yaw_forearm","skewness_roll_forearm","skewness_pitch_forearm","skewness_yaw_forearm","max_yaw_forearm","min_yaw_forearm","amplitude_yaw_forearm","stddev_roll_forearm",
"var_roll_forearm","avg_pitch_forearm","stddev_pitch_forearm","var_pitch_forearm","avg_yaw_forearm","stddev_yaw_forearm","var_yaw_forearm","user_name","cvtd_timestamp","skewness_roll_arm","skewness_pitch_arm","skewness_yaw_arm",
"max_roll_belt","max_picth_belt","min_roll_belt","min_pitch_belt","amplitude_roll_belt","amplitude_pitch_belt","var_total_accel_belt","avg_roll_belt","stddev_roll_belt","var_roll_belt","avg_pitch_belt","stddev_pitch_belt",
"var_pitch_belt","avg_yaw_belt","stddev_yaw_belt","var_yaw_belt","max_picth_arm","max_yaw_arm","min_yaw_arm","amplitude_pitch_arm","amplitude_yaw_arm","max_roll_dumbbell","max_picth_dumbbell","min_roll_dumbbell","min_pitch_dumbbell",
"amplitude_roll_dumbbell","amplitude_pitch_dumbbell","var_accel_dumbbell","avg_roll_dumbbell","stddev_roll_dumbbell","var_roll_dumbbell","avg_pitch_dumbbell","stddev_pitch_dumbbell","var_pitch_dumbbell","avg_yaw_dumbbell","stddev_yaw_dumbbell",
"var_yaw_dumbbell"," max_roll_forearm","max_picth_forearm","min_roll_forearm","min_pitch_forearm"," amplitude_roll_forearm","amplitude_pitch_forearm","var_accel_forearm","avg_roll_forearm","var_accel_arm","max_roll_forearm","amplitude_roll_forearm",
"max_roll_forearm","amplitude_roll_forearm","var_accel_arm") 
training2 <- training[!myvars]
str(training2)
names(training2)

t<-sapply(training2, function(x) sum(is.na(x)))
var.test(training2$classe,training2$X)

##Exploratory analysis
tapply(training2$X,training2$classe,summary)
tapply(training2$roll_belt,training2$classe,summary)
tapply(training2$accel_belt_z,training2$classe,summary)
library(RANN)

##Let's determine how many components
preProc <- preProcess(training2[,-57],method="pca",thresh=.90)
preProc
#try rpart to identify the key variables
modFit <- train(classe ~ ., method="rpart", data=training2)
prediction_tree <- cbind(testing$classe,predict(modFit,testing))
table(prediction_tree[,1],prediction[,2])
prediction_tree1 <- cbind(pml.testing$classe_num,predict(modFit,pml.testing))
table(prediction_tree1[,1],prediction_tree1[,2])

varImp(modFit, scale = FALSE)

print(modFit$finalModel)

#lda
modlda = train(classe ~ ., data=training2,method="lda")
warnings()
modlda$finalModel
prediction<-cbind(testing$classe,predict(modlda,testing))
table(prediction[,1],prediction[,2])
prediction1<-cbind(pml.testing$classe_num,predict(modlda,pml.testing))
table(prediction1[,1],prediction1[,2])
