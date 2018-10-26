###################################################################################################
#
# SDD 2018 Hackathon: ZüriWieNeu Classification Challenge
# shiny presentation app
#
###################################################################################################

library(shiny)
library(shinythemes)
library(jsonlite)
library(tidyverse)
library(reticulate)

# set working directory
setwd("~/Dropbox/Data_Projects/SDD_Hackathon")


###################################################################################################
# data download & preparation
###################################################################################################

# read url and convert to data.frame
url <- 'https://data.stadt-zuerich.ch/dataset/zueriwieneu_meldungen/resource/2fee5562-1842-4ccc-a390-c52c9dade90d/download/zueriwieneu_meldungen.json'
dataZWN <- fromJSON(txt=url)
dataZWN_df<-dataZWN$features$properties %>% as_tibble()

# rename categories
data_total <- mutate(dataZWN_df
                     , service_name = case_when(service_name == 'Abfall/Sammelstelle' ~ 'Cat1',
                                                service_name == 'Strasse/Trottoir/Platz' ~ 'Cat2',
                                                service_name == 'Signalisation/Lichtsignal' ~ 'Cat3',
                                                service_name == 'Grünflächen/Spielplätze' ~ 'Cat4',
                                                service_name == 'Beleuchtung/Uhren' ~ 'Cat5',
                                                service_name == 'Graffiti' ~ 'Cat6',
                                                service_name == 'VBZ/ÖV' ~ 'Cat7',
                                                service_name == 'Brunnen/Hydranten' ~ 'Cat8',
                                                TRUE ~ 'Error'))

###################################################################################################
# prediction function
###################################################################################################

# connect to the desired python installation
use_python('/usr/local/Cellar/python/3.7.0/bin/python3', required = TRUE)
py_config() # check if it has connected to the correct version

# source python script
source_python('python_presentation_model.py')

# tain model on full dataset
presentation_models <- train_model(data_total$detail, data_total$service_name)

# prediction function
predict_category <- function(text){
  pred <- py_predict(c(text, text), presentation_models)[1,]
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


# user interface
ui <- fluidPage(
  bootstrapPage(theme = shinytheme('superhero'),
    fluidRow(
      column(12, 
             align = 'center',
             titlePanel('Swiss Digital Day 2018 Hackathon'),
             h3('Züri Wie Neu Challenge'), br(), br(),
             img('', src = 'Logo_128.png'), br(), br(),
             textInput(inputId = 'text', label = 'Bitte eine Beschreibung des Schadens angeben'),
             textOutput('cat')
      )
    )
  )
)

# server
server <- function(input, output){
  output$cat <- renderText(if(input$text == ''){''}else{predict_category(input$text)})
}

# run app
shinyApp(ui, server)
