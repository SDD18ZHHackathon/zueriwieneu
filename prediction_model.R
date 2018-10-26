###################################################################################################
#
# SDD 2018 Hackathon: ZüriWieNeu Classification Challenge
# prediction functions
#
###################################################################################################

library(jsonlite)
library(tidyverse)
library(tm)
library(SnowballC)
library(Matrix)
library(caret)

# set working directory
setwd("~/Dropbox/Data_Projects/SDD_Hackathon")

# read url and convert to data.frame
url <- 'https://data.stadt-zuerich.ch/dataset/zueriwieneu_meldungen/resource/2fee5562-1842-4ccc-a390-c52c9dade90d/download/zueriwieneu_meldungen.json'
dataZWN <- fromJSON(txt=url)
dataZWN_df<-dataZWN$features$properties %>% as_tibble()


###################################################################################################
# location prediction function
###################################################################################################

loc_prob <- function(x, y, n = 10, positions){
  # for a given location (x,y) the function searches the closest n neighbors in the positions table
  # and predicts the category by a weighted average
  
  # output vector
  cats <- paste0('Cat', 1:8)
  output <- rep(0, length(cats))
  names(output) <- cats
  
  # calc distances
  dist <- sqrt((positions$e - x)^2 + (positions$n - y)^2)
  close_n_ind <- order(dist)[1:n]
  close_n_dist <- 1/(dist[close_n_ind] + 1)
  
  # fill output vector
  total_weight <- sum(close_n_dist)
  close_n_cats <- positions[close_n_ind,]$service_name
  for(cat in unique(close_n_cats)){
    num <- sum(close_n_dist[close_n_cats == cat])
    prob <- num / total_weight
    output[names(output) == cat] <- prob
  }
  
  # add max distance
  output['max_dist'] <- max(1/close_n_dist)
  names(output) <- paste0('loc_', names(output))
  
  return(output)
}


###################################################################################################
# description prediction function 1
###################################################################################################

roccio_alg <- function(query,doc.list,spec.input='ntc',cate){
  
  my.docs <- VectorSource(c(doc.list, query))
  q.len <- length(query)
  
  my.corpus <- Corpus(my.docs)
  my.corpus <- tm_map(my.corpus, enc2utf8)
  my.corpus <- tm_map(my.corpus, removePunctuation)
  my.corpus <- tm_map(my.corpus, removeNumbers)
  my.corpus <- tm_map(my.corpus, stripWhitespace)
  my.corpus <- tm_map(my.corpus, content_transformer(tolower))
  my.corpus <- tm_map(my.corpus, removeWords, stopwords("german"))
  my.corpus <- tm_map(my.corpus, stemDocument, language = "german")
  
  term.doc.matrix.stm <- TermDocumentMatrix(my.corpus)
  
  ui = unique(term.doc.matrix.stm$j)
  term.doc.matrix.stm.new = term.doc.matrix.stm[,ui]
  n.l <- ncol(term.doc.matrix.stm.new)
  
  tfidf.matrix <- Matrix(as.matrix(weightSMART(term.doc.matrix.stm.new, spec = spec.input)), sparse = TRUE)
  
  query.vector <- tfidf.matrix[, ((n.l-(q.len-1)):n.l)]
  tfidf.matrix <- tfidf.matrix[, 1:(n.l-1)]
  
  
  cate_Abfall_Sammelstelle <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat1"],1,'mean')
  cate_Strasse_Trottoir_Platz <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat2"],1,'mean')
  cate_Signalisation_Lichtsignal <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat3"],1,'mean')
  cate_Grünflächen_Spielplätze <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat4"],1,'mean')
  cate_Beleuchtung_Uhren <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat5"],1,'mean')
  cate_Graffiti <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat6"],1,'mean')
  cate_VBZ_ÖV <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat7"],1,'mean')
  cate_Brunnen_Hydranten <- apply(tfidf.matrix[,cate[ui[1:(n.l-q.len)]]=="Cat8"],1,'mean')
  
  
  cate.mat <- cbind(cate_Abfall_Sammelstelle,cate_Strasse_Trottoir_Platz,
                    cate_Signalisation_Lichtsignal,cate_Grünflächen_Spielplätze,
                    cate_Beleuchtung_Uhren,cate_Graffiti,cate_VBZ_ÖV,cate_Brunnen_Hydranten)
  
  doc.scores <- t(query.vector) %*% cate.mat
  
  doc.scores.proz<-prop.table(as.matrix(doc.scores),1)
  
  colnames(doc.scores.proz) <- c('desc_Cat1','desc_Cat2',
                                 'desc_Cat3','desc_Cat4',
                                 'desc_Cat5','desc_Cat6',
                                 'desc_Cat7','desc_Cat8')
  #
  # colnames(doc.scores.proz) <- c('Abfall/Sammelstelle','Strasse/Trottoir/Platz',
  #                                'Signalisation/Lichtsignal','Grünflächen/Spielplätze',
  #                                'Beleuchtung/Uhren','Graffiti','VBZ/ÖV','Brunnen/Hydranten')
  
  results.df <- data.frame(as.matrix(doc.scores.proz))
  results.df.out <- results.df
  return(results.df.out)
}


###################################################################################################
# description prediction function 2
###################################################################################################

# connect to the desired python installation
library(reticulate)
use_python('/usr/local/Cellar/python/3.7.0/bin/python3', required = TRUE)
py_config() # check if it has connected to the correct version

# source python script
source_python('python_desc_model.py')


###################################################################################################
# prepare train & test data
###################################################################################################

# create partitions (40%, 40%, 20%)
set.seed(123)
partition1 <- createDataPartition(dataZWN_df$service_name, p = 0.8)
partition2 <- createDataPartition(dataZWN_df[partition1$Resample1,]$service_name, p = 0.5)

# prepare data
data_total <- mutate(dataZWN_df
                     , service_name = case_when(service_name == 'Abfall/Sammelstelle' ~ 'Cat1',
                                                service_name == 'Strasse/Trottoir/Platz' ~ 'Cat2',
                                                service_name == 'Signalisation/Lichtsignal' ~ 'Cat3',
                                                service_name == 'Grünflächen/Spielplätze' ~ 'Cat4',
                                                service_name == 'Beleuchtung/Uhren' ~ 'Cat5',
                                                service_name == 'Graffiti' ~ 'Cat6',
                                                service_name == 'VBZ/ÖV' ~ 'Cat7',
                                                service_name == 'Brunnen/Hydranten' ~ 'Cat8',
                                                TRUE ~ 'Error')
                     , interface = case_when(interface_used == 'Web interface' ~ 'Web'
                                             , interface_used == 'iOS' ~ 'iOS'
                                             , interface_used == 'Android' ~ 'Android'
                                             , interface_used == 'iPhone' ~ 'iOS'
                                             , interface_used == 'iPad' ~ 'iOS'
                                             , interface_used == 'iPod touch' ~ 'iOS'))
dataset1 <- data_total[partition1$Resample1[partition2$Resample1],]
dataset2 <- data_total[partition1$Resample1[-partition2$Resample1],]
dataset3 <- data_total[-partition1$Resample,]

# save and load
#save(dataset1, dataset2, dataset3, file = 'dataset_split.Rda')
#load('dataset_split.Rda')


###################################################################################################
# train & predict
###################################################################################################

# train layer 1 models
python_models <- train_model(dataset1$detail, dataset1$service_name)

# predict on dataset 2 and 3
pred2_loc <- t(mapply(loc_prob, dataset2$e, dataset2$n, MoreArgs = list(n = 20, positions = dataset1)))
pred3_loc <- t(mapply(loc_prob, dataset3$e, dataset3$n, MoreArgs = list(n = 10, positions = dataset1)))
pred2_desc <- roccio_alg(dataset2$description, doc.list = dataset1$description, cate = dataset1$service_name)
pred3_desc <- roccio_alg(dataset3$description, doc.list = dataset1$description, cate = dataset1$service_name)
pred2_py <- predict(dataset2$description, python_models)
pred3_py <- predict(dataset3$description, python_models)

pred2 <- as_tibble(cbind(as_tibble(pred2_loc), as_tibble(pred2_desc))) %>% 
  add_column(interface = as.factor(dataset2$interface), py_cat = as.factor(pred2_py))

pred3 <- as_tibble(cbind(as_tibble(pred3_loc), as_tibble(pred3_desc))) %>% 
  add_column(interface = as.factor(dataset3$interface), py_cat = as.factor(pred3_py))

# train layer 2 model
control <- trainControl(method="cv"
                        , number=2
                        , allowParallel = FALSE
                        , verboseIter = TRUE
                        , classProbs = TRUE
                        )

model <- train(y = as.factor(dataset2$service_name)
               , x = as.data.frame(pred2)
               , method="ranger"
               , metric="Kappa"
               , trControl=control
               )

print(model)
plot(model)


###################################################################################################
# model validation
###################################################################################################

# layer 1 model validation
cat_loc <- max.col(pred2_loc[,1:8])
cat_desc <- max.col(pred2_desc)
table(pred2_py, dataset2$service_name) # confusion matrix
sum(diag(as.matrix(table(pred2_py, dataset2$service_name))))/sum(as.matrix(table(pred2_py, dataset2$service_name))) # accuracy

# final model validation
final_pred <- predict(model, pred3, type = 'raw')
final_pred_cat <- as_tibble(predict(model, pred3, type = 'prob')) %>% add_column(y_true = dataset3$service_name)
in_top_n <- apply(final_pred_cat, 1, function(x) as.numeric(substr(x[9],4,4)) %in% order(x[1:8], decreasing = TRUE)[1:3])
sum(in_top_n)/length(in_top_n) # 90% accuracy that correct answer is in the top 3 predictions

table(final_pred, dataset3$service_name) # confusion matrix
sum(diag(as.matrix(table(final_pred, dataset3$service_name))))/sum(as.matrix(table(final_pred, dataset3$service_name))) # accuracy


###################################################################################################
# prediction function (for final presentation)
###################################################################################################

presentation_models <- train_model(data_total$detail, data_total$service_name)

predict_category <- function(text){
  pred <- predict(c(text, text), presentation_models)[1,]
  cat <- which.max(pred)
  return(paste(case_when(cat == 1 ~ 'Abfall/Sammelstelle',
                   cat == 2 ~  'Strasse/Trottoir/Platz',
                   cat == 3 ~  'Signalisation/Lichtsignal',
                   cat == 4 ~  'Grünflächen/Spielplätze',
                   cat == 5 ~  'Beleuchtung/Uhren',
                   cat == 6 ~  'Graffiti',
                   cat == 7 ~  'VBZ/ÖV',
                   cat == 8 ~  'Brunnen/Hydranten',
                   TRUE ~ 'No comment on this one...'),sprintf('%1.1f%%',100*max(pred))))
}
