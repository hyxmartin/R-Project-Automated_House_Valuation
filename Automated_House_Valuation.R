# -------------------------------
# Author: Yuxiang Hu
# Time: 1/13/2017
# -------------------------------

library(FNN)
library(gplots)
library(ggplot2)
library(plyr)

# Function to import csv
import.csv <- function(filename) {
    return(read.csv(filename, sep = ",", header = TRUE))
}

# Function to export csv
write.csv <- function(ob, filename) {
    write.table(ob, filename, quote = FALSE, sep = ",", row.names = FALSE)
}

# Function to remove special characters
trim <- function(x) gsub("^\\s+|\\s+$", "", x)

# Function to determine missing symbol
is.missing.symbol <- function(x) {
    if (nchar(trim(x)) == 0) {
        missing = 1
    } else {
        missing = 0
    }
    return(missing)
}




brief <- function(df) {
    
    # those two list store two indexs, one for real and another for symbolic attributes
    real.index <- NULL
    symbol.index <- NULL
    
    num_row <- nrow(df)  # count number of rows 
    num_att <- ncol(df)  # count number of columns 
    
    for (ii in 1:num_att) {
        this.att <- df[, ii]
        if (is.numeric(this.att)) {
            real.index <- c(real.index, ii)
        }
        if (is.factor(this.att)) {
            symbol.index <- c(symbol.index, ii)
        }
    }
    
    cat("This data set has", num_row, "rows,", num_att, "attributes")  # write a line here: use cat function to print the "This data set has xxx row, xxx attributes
    
    real.out <- NULL  # this data frame store information for real valued attributes
    for (index in real.index) {
        this.att <- df[, index]  # extract a specific column
        att_name <- colnames(df)[index]  # get attribute name using colnames function
        num_missing <- sum(is.na(this.att))  # count number of missing values using is.na function 
        Min <- round(min(this.att), 2)  # min
        Max <- round(max(this.att), 2)  # max
        Mean <- round(mean(this.att), 2)  # mean
        Median <- round(median(this.att), 2)  # median
        Sdev <- round(sd(this.att), 2)  # standard deviation
        Var <- round(var(this.att), 2)  # variance
        this.line <- data.frame(Attribute_ID = index, Attribute_Name = att_name, Missing = num_missing, 
            Min, Max, Mean, Median, Sdev, Variance = Var) # assemble into a line
        real.out <- rbind(real.out, this.line) # concatenate to get a data frame
    }
    
    cat("real valued attributes\n")
    cat("======================\n")
    print(real.out)
    
    # gather stats for symbolic attributes
    
    symbol.out <- NULL #this data frame store information for real valued attributes
    max_MCV = 3
    for (index in symbol.index) {
        this.att <- df[, index]
        att_name <- colnames(df)[index]  # get attribute name 
        
        #count number of missing values in symbolic attribute
        num_missing <- length(this.att[this.att == ''])
        non_missing_id <- which(unlist(lapply(this.att, is.missing.symbol)) == 0)        
        this.att <- this.att[non_missing_id]
  
        #count MCV
        arity <- nlevels(this.att[this.att!='',drop=TRUE]) 
        num_MCV <- min(max_MCV, arity)
        count.tbl <- as.data.frame(table(this.att))
        sorted <- count.tbl[order(-count.tbl$Freq), ][1:num_MCV, ]
        MCV_str <- ""
        for (kk in 1:nrow(sorted)) {
            MCV_value <- sorted[kk, c("this.att")]  # MCV string
            MCV_count <- sorted[kk, c("Freq")]  # MCV count
            this_str <- paste(MCV_value, "(", MCV_count, ")", sep = "")
            MCV_str <- paste(MCV_str, this_str, sep = " ")
        }
        
        
        this.line <- data.frame(Attribute_ID = index, Attribute_Name = att_name, Missing = num_missing, 
            arity, MCVs_counts = MCV_str)
        symbol.out <- rbind(symbol.out, this.line)
    }
    cat("symbolic attributes\n")
    cat("===================\n")
    print(symbol.out)
    
}

df<-import.csv('house_no_missing.csv')  # import csv from working directory
brief(df)  # call brief function




# utility function for import from csv file
import.csv <- function(filename) {
  return(read.csv(filename, sep = ",", header = TRUE))
}

# utility function for export to csv file
write.csv <- function(ob, filename) {
  write.table(ob, filename, quote = FALSE, sep = ",", row.names = FALSE)
}

# Connect-the-dots model that learns from train set and is being tested using test set
# Assumes the last column of data is the output dimension
get_pred_dots <- function(train, test){
  nf <- ncol(train)
  input <- train[,-nf]
  query <- test[,-nf]
  my.knn <- get.knnx(input, query, k=2) # Get two nearest neighbors
  nn.index <- my.knn$nn.index
  pred <- rep(NA, nrow(test))
  for (ii in 1:nrow(test)){
    y1 <- train[nn.index[ii, 1], nf]
    y2 <- train[nn.index[ii, 2], nf]
    pred[ii] = (y1 + y2) / 2
  }
  return(pred)  
}

# Linear model
# Assumes the last column of data is the output dimension
get_pred_lr <- function(train,test){
  nf <- ncol(train)  # number of features of train/test data
  output <- colnames(train)[nf]
  lm.model <- lm(as.formula(paste(output, "~ .")), data = train)  # Use variable in lm formula, it needs to convert by formula function.
  pred <- predict(lm.model, test)  # make prediction using the model check out the predict function for lm 
  return(pred)
}

# Default predictor model
# Assumes the last column of data is the output dimension
# Default predictor simply takes the mean of the output of training set, 
# use it as the predictor for all test points.  
# When doing modeling, it is always  a good idea to compare to the default model, 
# so that we could know how much improvement we get against the baseline.
get_pred_default <- function(train,test){
  nf <- ncol(train)
  pred <- rep(mean(train[, nf]), nrow(test))
}

# do_cv
# df is the dataframe of input
# output is the output column name
# k is k folds cross-validation
# model is the model selection
do_cv <- function(df, output, k, model) {
  nn <- nrow(df)  # number of data points
  nf <- ncol(df)  # number of features
  df <- df[sample(1:nn), ]  # randomize data
  df <- df[,c(which(colnames(df) != output), which(colnames(df) == output))]  # Move output to the last column
  folds <- get_folds(nn, k)
  score <- rep(NA, length(folds))  # create a list to hold the mse for each folds
  for (ii in 1:length(folds)) {
    test.index <- folds[[ii]]  # extract test index 
    train.data <- df[-test.index, ]  # assemble training data
    test.data <- df[test.index, ]  # assemble test data
    pred <- model(train.data, test.data)
    true <- test.data[, c(which(colnames(df) == output))]  # extract true values from test set
    mse <- 1 / length(pred) * sum((pred - true) ^ 2)  # computer mean squared error
    score[ii] <- mse  # save the score    
  }
  return(score)
}

get_folds <- function(nn, k) {
  index <- seq(1, nn)
  rand.index <- sample(index, nn)
  group <- seq_along(rand.index)%%k
  chunk <- split(rand.index, group)
  return(chunk)
}


# Load data and transform Crime_Rate to Log(Crime_Rate)
df <- import.csv("house_no_missing.csv")
my.data <- df[, c("Crime_Rate", "house_value")]
my.data <- transform(my.data, 
                Crime_Rate = log(Crime_Rate)
)

# Parameter for K Folds Cross Validation
k <- 10
# Run for Connect-the-dots model
get_pred_dots_score <- do_cv(my.data, "house_value", k, get_pred_dots)
ggplot(data.frame(x = 1 : k, y = get_pred_dots_score), aes(x=x, y=y)) + 
  geom_bar(stat="identity") + 
  geom_hline(yintercept = mean(get_pred_dots_score), color="red") +
  xlab("number of K fold") + 
  ylab("score") +
  ggtitle("Connect-the-dots model")

# Run for linear model
get_pred_lr_score <- do_cv(my.data, "house_value", k, get_pred_lr)
ggplot(data.frame(x = 1 : k, y = get_pred_lr_score), aes(x=x, y=y)) + 
  geom_bar(stat="identity") + 
  geom_hline(yintercept = mean(get_pred_lr_score), color="red") +
  xlab("number of K fold") + 
  ylab("score") +
  ggtitle("linear model")

# Run for default model
get_pred_default_score <- do_cv(my.data, "house_value", k, get_pred_default)
ggplot(data.frame(x = 1 : k, y = get_pred_default_score), aes(x=x, y=y)) + 
  geom_bar(stat="identity") + 
  geom_hline(yintercept = mean(get_pred_default_score), color="red") +
  xlab("number of K fold") + 
  ylab("score") +
  ggtitle("default model")

# Prepare for plot
model.name <- c("Connect-the-dots model","Linear model","Default Predictor model")
model.mean <- c(mean(get_pred_dots_score), mean(get_pred_lr_score), mean(get_pred_default_score))
t.test <- t.test(get_pred_dots_score, conf.level = 0.95)
ci.l <- c(t.test$conf.int[1])
ci.h <- c(t.test$conf.int[2])
t.test <- t.test(get_pred_lr_score, conf.level = 0.95)
ci.l <- c(ci.l, t.test$conf.int[1])
ci.h <- c(ci.h, t.test$conf.int[2])
t.test <- t.test(get_pred_default_score, conf.level = 0.95)
ci.l <- c(ci.l, t.test$conf.int[1])
ci.h <- c(ci.h, t.test$conf.int[2])
compare.scores <- data.frame(model.name, model.mean, ci.l, ci.h)

# Plot barchart
bplot <- barplot2( compare.scores$model.mean,  # Data (bar heights) to plot  
          beside = TRUE,  # Plot the bars beside one another; default is to plot stacked bars  
          names.arg = compare.scores$model.name,  #Names for the bars  
          col = c("lightblue", "mistyrose", "lightcyan"),  # Color of the bars  
          border ="black",  # Color of the bar borders  
          main = c("Model comparison"),  # Main title for the plot  
          xlab = "Model",  # X-axis label  
          ylab = "Score",  # Y-axis label  
          font.lab = 2,  # Font to use for the axis labels: 1=plain text, 2=bold, 3=italic, 4=bold italic  
          plot.ci = TRUE,  # Plot confidence intervals  
          ci.l = compare.scores$ci.l,  # Lower values for the confidence interval  
          ci.u = compare.scores$ci.h,  # Upper values for the confidence interval  
          plot.grid = TRUE)  # Plot a grid  
legend(   "top",  # Add a legend to the plot  
          legend = compare.scores$model.name,  # Text for the legend  
          fill = c("lightblue", "mistyrose", "lightcyan"),  # Fill for boxes of the legend  
          bg = "white")  # Background for legend box 
text(bplot, compare.scores$model.mean, labels = compare.scores$model.mean, pos = 3)
