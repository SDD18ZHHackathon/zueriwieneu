###################################################################################################
#
# SDD 2018 Hackathon: ZüriWieNeu Classification Challenge
# shiny presentation app
#
###################################################################################################

library(shiny)
library(shinythemes)

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
