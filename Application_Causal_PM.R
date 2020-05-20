#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

#Import packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(c("shiny", "shinydashboard", "rhandsontable", "DT",
                 "tidyverse", "paletteer", "magrittr",
                 "eulerr", "ggpubr", "egg", "ggstance", "ggalluvial",
                 "factoextra", "FactoMineR", "ggrepel",
                 "lava", "nnet", "DescTools", "tictoc", "patchwork"),
               character.only = TRUE)

#Define some color vectors
colors_genes <- c(paletteer_d("nord::lumina"))

colors_drugs <- c(paletteer_d("ggsci::default_uchicago"))[c(1,3,4,5,9)]

#colors_methods <- c(paletteer_d("nord::red_mountain"))
colors_methods <- c("True"= "#376597FF", "Std"="#928F6BFF", "IPW"="#CCAF69FF",
                    "TMLE"="#556246FF", "Naive"="#8F5144FF")

# Define UI for application that draws a histogram
ui <- dashboardPage(
   
   # Application title
   dashboardHeader(title = "Causal Precision Medicine",
                   titleWidth = 280),
   ## Sidebar content
   dashboardSidebar(
     width = 280,
     sidebarMenu(
       menuItem("Genes", tabName = "genes", icon = icon("dna")),
       menuItem("Precision medicine", tabName = "algorithm", icon = icon("random")),
       menuItem("Treatments", tabName = "treatments", icon = icon("pills")),
       menuItem("Simulation", tabName = "simulation", icon = icon("cogs")),
       menuItem("Estimates", tabName = "estimates", icon = icon("laptop")),
       menuItem("Background", tabName = "background", icon = icon("question"))
     )
   ),
   dashboardBody(
     tabItems(
       # First tab content
       tabItem(tabName = "genes",
               h2("1. Genes"),
               h4("First we have to define what are the genes that will be used in our precision medicine algorithm"),
               fluidRow(
                 box(
                   title = "1.1. Set the number of genes in the precision medicine algorithm",
                   solidHeader = FALSE, status = "primary", width=6,
                   sliderInput("slider_genes", "Number of genes:", 1, 5, 3)
                 ),
                 box(
                   withMathJax(),
                   title = "1.2 Change default gene names (initially \\(C_{i}\\)) and prevalence",
                   solidHeader = FALSE, status = "primary", width=6,
                   rHandsontableOutput("genes_prevalence"),
                   br(),
                   actionButton(inputId = "plot_prevalence", label = "Update data and create plot")
                 )
                 
               ),
               fluidRow(
                 box(
                   title = "1.3. Theoretical proportion of combinations of genes",
                   solidHeader = FALSE, status = "primary", width=12,
                   plotOutput("euler_plot")
                 )
               )
       ),
       
       #Second tab content
       tabItem(tabName = "algorithm",
               h2("2. Precision medicine algorithm"),
               h4("Now we define the personalized treatment recommanded to each category of patients"),
               fluidRow(
                 box(
                   title = "2.1. Set the number of personalized treatments in the precision medicine algorithm",
                   solidHeader = FALSE, status = "primary", width=6,
                   sliderInput("slider_treatments", "Number of treatment:", 1, 5, 3)
                 ),
                 box(
                   title = "2.2 Define treatments of interest",
                   solidHeader = FALSE, status = "primary", width=6,
                   uiOutput("treatment_choice"),
                   br(),
                   actionButton(inputId = "start_assignment", label = "Start treatment assignment")
                 )
               ),
               fluidRow(
                 box(
                   title = "2.3. For each group of patients, define the precision medicine treatment (\\(K_{PM}\\)) and the distribution probability of observed treatments \\(P(K=k_i)=P\\_k\\_i\\). Please note that \\(k_{0}\\) corresponds to control patients, here interpreted as untreated",
                   solidHeader = FALSE, status = "primary", width=12,
                   rHandsontableOutput("treatment_assignment_hot_output")
                 )
               ),
               fluidRow(
                 box(
                   title = "2.4. Proportions of personalized treatments \\(K_{PM}=r(C)\\)",
                   solidHeader = FALSE, status = "primary", width=6,
                   plotOutput("treatment_assignment_hot_plot_render")
                 ),
                 box(
                   title = "2.5. Proportions of observed treatments (\\(K\\))",
                   solidHeader = FALSE, status = "primary", width=6,
                   plotOutput("treatment_observed_hot_plot_render")
                 )
               )
       ),
       
       # Third tab content
       tabItem(tabName = "treatments",
               h2("3. Treatments and influence of biomarkers"),
               h4("Here we define how treatments behave depending on biomarkers, with R package lava"),
               fluidRow(
                 uiOutput("tbox"),
                 box(title = "Summary",
                     solidHeader = FALSE, status = "primary", width=6,
                     #tableOutput("test_table"),
                     plotOutput("treatment_params_plot"))
               )
       ),
       
       # Fourth tab content
       tabItem(tabName = "simulation",
               h2("4. Simulation"),
               h4("Here we generate data with R package lava and observe distributions."),
               fluidRow(
                 box(
                   title = "4.1. Generate data",
                   solidHeader = FALSE, status = "primary", width=12,
                   sliderInput("slider_size_super_pop", "Size super-population:", 5000, 20000, step = 5000, value=10000),
                   br(),
                   actionButton(inputId = "start_simulation", label = "Simulate data")
                 )
               ),
               fluidRow(
                 box(#title = "4.2 Summary plot of simulated data (only eligible patients)",
                     title = "4.2 Summary plot of simulated data (only eligible patients). \\(Y_{k_i}\\) is the response to drug \\(k_{i}\\) and \\(Y_{PM}\\) to personalized treatment.",
                     solidHeader = FALSE, status = "primary", width=12,
                     uiOutput("simulated_data_select"),
                     plotOutput("simulated_lava_plot", height = 700))
               ),
               fluidRow(
                 box(title = "4.3 Table of simulated data (all patients)",
                     solidHeader = FALSE, status = "primary", width=12,
                     dataTableOutput("sim_table"))
               )
       ),
       
       # Fourth tab content
       tabItem(tabName = "estimates",
               h2("5. Compute estimates"),
               h4("We compute estimates for \\(CE_{1}\\), \\(CE_{2}\\) and \\(CE_{3}\\) in all sub-cohorts"),
               fluidRow(
                 box(
                   title = "5.1. Sampling parameters",
                   solidHeader = FALSE, status = "primary", width=12,
                   sliderInput("slider_number_sub_coh", "Number of sub-cohorts for sampling:", 100, 2000, step = 100, value=1000),
                   sliderInput("slider_size_sub_coh", "Size of sub-cohorts:", 50, 1000, step = 50, value=200),
                   br(),
                   actionButton(inputId = "start_estimates", label = "Compute causal estimates")
                 )
               ),
               fluidRow(
                 box(title = "5.2 Plot parameters",
                     solidHeader = FALSE, status = "primary", width=4,
                     selectizeInput(
                       'estimates_select', 'Causal effect:',
                       choices = c("CE1", "CE2", "CE3"),
                       select="CE1"
                     ),
                     selectizeInput(
                       'estimates_perf', 'Deviation metric:',
                       choices = c("MAE", "RMSE"),
                       select="MAE"
                     ),
                     selectizeInput(
                       'estimates_PCA_var', 'PCA variables:',
                       choices = c("Genes", "K_PM", "Both"),
                       select="Genes"
                     ),
                     selectizeInput(
                       'estimates_PCA_enrich', 'Data types:',
                       choices = c("Proportions", "Enrichment", "Both"),
                       select="Proportions"
                     ),
                     selectizeInput(
                       'estimates_PCA_color', 'Color of points:',
                       choices = c("Absolute CE", "Deviation from true CE"),
                       select="Absolute CE"
                     )),
                 
                 box(title = "5.3 Summary plot of estimates",
                     solidHeader = FALSE, status = "primary", width=8,
                     plotOutput("estimates_plot", height = 600),
                     plotOutput("estimates_plot_PCA", height = 300))
               ),
               fluidRow(
                 box(title = "5.3 Table of estimates data",
                     solidHeader = FALSE, status = "primary", width=12,
                     dataTableOutput("estimates_table"))
               )
       ),
       
       #Fifth tab content
       tabItem(tabName = "background",
               h2("Background"),
               h4("Please refer to the original article for details abot the objectives and methods of this work. Below is the schematic representation of the target trials and the subsequent definition of \\(CE_{1}\\), \\(CE_{2}\\) and \\(CE_{3}\\)"),
               imageOutput("photo")
       )
     )
   )
)

# Define server logic
server <- function(input, output) {
  
  #Here we define the editable table to rename genes and define their prevalence in the super population
  output$genes_prevalence <-  renderRHandsontable({
    data.frame(
      Genes = paste0("C_", 1:input$slider_genes),
      Prevalence = 0.3,
      stringsAsFactors = FALSE
    ) %>%
      rhandsontable
  })
  
  #The following vector defines consistent colors for the genes
  data_colors_genes <- eventReactive(input$plot_prevalence, {
    genes_prevalence <- input$genes_prevalence
    if (!is.null(genes_prevalence)){
      genes_prevalence %<>% hot_to_r
    }
    final_colors <- colors_genes[1:nrow(genes_prevalence)]
    names(final_colors) <- genes_prevalence$Genes %>% as.character
    final_colors
  })
  
  #This reactive dataframe is used to compute the theoretical proportions of mutations
  data_gene_prevalence_combined <- eventReactive(input$plot_prevalence, {
    genes_prevalence <- input$genes_prevalence
    if (!is.null(genes_prevalence)){
      genes_prevalence %<>% hot_to_r
    }
    combined <- expand.grid(replicate(nrow(genes_prevalence), 0:1, simplify = FALSE))
    colnames(combined) <- as.character(genes_prevalence$Genes)
    probs <- genes_prevalence$Prevalence
    probs_comb <- rep(1, nrow(combined))
    for (i in 1:ncol(combined)){
      probs_comb <- probs_comb*(combined[,i]*probs[i] + (1-combined[,i])*(1-probs[i]))
    }
    combined %<>% mutate(`No mutations`=if_else(rowSums(.)==0, 1, 0),
                         Probability=probs_comb)
    combined
  })
  
  #Plot of relative proportions of combinations of mutations
  output$euler_plot <- renderPlot({
    dat <- data_gene_prevalence_combined()
    euler(select(dat, -Probability), weights = dat$Probability, shape = "ellipse") %>%
      plot(quantities=list(type="percent", font=5),
           fills = list(fill = data_colors_genes(),
                        alpha = 0.5))
  })
  
  #Define the treatments choice
  output$treatment_choice <- renderUI({
    
    selectizeInput('list_treatments', label="Defaut drugs can be replaced by custom names",
                   multiple = TRUE,
                   choices = paste0("k_", 1:input$slider_treatments),
                   selected = paste0("k_", 1:input$slider_treatments),
                   options = list(create=TRUE, maxItems = input$slider_treatments))
  })
  
  
  #Define the treatment algorithm
  treatment_assignment_hot <- eventReactive(input$start_assignment, {
    list_treatments <- input$list_treatments
    dat <- data_gene_prevalence_combined() %>%
      filter(`No mutations`!=1) %>%
      select(-`No mutations`) %>%
      mutate(K_PM=factor(list_treatments[1], levels=list_treatments))
    dat[paste0("P_",c("k_0", list_treatments))] <- 1/(length(list_treatments)+1)
    rhandsontable(dat, readOnly = TRUE) %>%
      hot_col(c("K_PM", "P_k_0", paste0("P_",list_treatments)), readOnly = FALSE)
  })
  
  output$treatment_assignment_hot_output <-  renderRHandsontable({
    treatment_assignment_hot()
  })
  
  treatment_assignment_hot_plot <- eventReactive(input$treatment_assignment_hot_output, {
    input$treatment_assignment_hot_output %>%
      hot_to_r %>%
      group_by(K_PM) %>%
      summarise(Percentage=sum(Probability)) %>%
      add_row(K_PM="None", Percentage=1-sum(.$Percentage)) %>%
      mutate(ymax=cumsum(Percentage),
             label=paste0(K_PM, "\n", as.character(100*Percentage), "%")) %>%
      mutate(ymin=c(0, head(ymax, n=-1)),
             labelPosition=(ymax+ymin)/2)
  })
  
  output$treatment_assignment_hot_plot_render <- renderPlot({
    treatment_assignment_hot_plot() %>%
      ggplot(aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=K_PM)) +
      geom_rect() +
      scale_fill_manual(values=data_colors_drugs()) +
      coord_polar(theta="y") +
      geom_text( x=3.5, aes(y=labelPosition, label=label), size=6, color="white") +
      xlim(c(2, 4)) + ylim(c(0,1)) +
      theme_void() +
      theme(legend.position = "none")
  })
  
  #And the corresponding colors
  #The following vector defines consistent colors for the genes
  data_colors_drugs <- eventReactive(input$start_assignment, {
    list_treatments <- input$list_treatments
    final_colors <- colors_drugs[1:length(list_treatments)]
    names(final_colors) <- list_treatments %>% as.character
    final_colors <- c(final_colors, "None"="grey20", "k_0"="grey80")
  })
  
  #Reprocess the assignment table for later use
  treatment_assignment_hot_reprocess <- eventReactive(input$treatment_assignment_hot_output, {
    input$treatment_assignment_hot_output %>%
      hot_to_r %>%
      mutate(Sum_P=rowSums(select(., starts_with("P_")))) %>%
      mutate_at(dplyr::vars(starts_with("P_")), function(x) x/(.$Sum_P)) %>%
      select(-Sum_P)
  })
  
  output$treatment_observed_hot_plot_render <- renderPlot({
    treatment_assignment_hot_reprocess() %>%
      mutate_at(dplyr::vars(starts_with("P_")), function(x) x*(.$Probability)) %>%
      select(starts_with("P_")) %>%
      rename_all(function(x) substr(x,3, stop=1000)) %>%
      colSums %>% as.data.frame %>%
      rename(Percentage=".") %>% rownames_to_column(var="K") %>%
      add_row(K="None", Percentage=1-sum(.$Percentage)) %>%
      mutate(ymax=cumsum(Percentage),
             label=paste0(K, "\n", as.character(100*round(Percentage,digits = 3)), "%")) %>%
      mutate(ymin=c(0, head(ymax, n=-1)),
             labelPosition=(ymax+ymin)/2) %>%
      ggplot(aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=K)) +
      geom_rect() +
      scale_fill_manual(values=data_colors_drugs()) +
      coord_polar(theta="y") +
      geom_text( x=3.5, aes(y=labelPosition, label=label), size=6, color="white") +
      xlim(c(2, 4)) + ylim(c(0,1)) +
      theme_void() +
      theme(legend.position = "none")
  })
  
  #Define the panels for treatments effects
  output$tbox <- renderUI({
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    tabs <- sapply(c("k_0", input$list_treatments), function(i) {
      
      tabPanel(title = paste0("Treat. ", i),
               h4(if_else(i=="k_0",
                          "Generic parameters (CAUTION, k_0 correspond to the control/untreated):",
                          "Generic parameters:")),
               sliderInput(paste0("slider_treat_Int_",i), paste0("Intercept for response to ", i), -50, 50, step = 5, value=if_else(i=="k_0", 10, 0)),
               sliderInput(paste0("slider_treat_Cov_",i), paste0("Variance for response to ", i), 0, 100, step = 5, value=25),
               br(),
               h4("Influence of biomarkers:"),
               sliderInput(paste0("slider_treat_",i,"_Aggressiveness"), paste0("Influence of Aggressiveness on response to ", i), 0, 20, step = 5, value=10),
               lapply(list_genes, function(j) {
                 sliderInput(inputId = paste0("slider_treat_",i, "_", j),
                             label = paste0("Influence of ", j, " on response to ", i),
                             min = -30, max = 20, step = 5, value=if_else(i=="k_0", 0, -10))
               })
      )
    },
    simplify = FALSE, USE.NAMES = FALSE)
    
    args = c(tabs, list(id = "box",
                        width=6,
                        title=tagList(icon("pills"), "Select a treatment")))
    do.call(tabBox, args)
  })
  
  #Process the drugs paramaters
  treatment_params <- reactive({
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments <- c("k_0",input$list_treatments)
    df <- data.frame(matrix(ncol=length(list_treatments),nrow=length(list_genes),
                            dimnames=list(list_genes, list_treatments)))
    for (g in c("Aggressiveness", list_genes)){
      for (t in list_treatments){
        df[g,t] <- input[[paste0("slider_treat_", t, "_", g)]]
      }
    }
    df
  })
  
   treatment_inter <- reactive({
     list_treatments <- c("k_0",input$list_treatments)
     df <- data.frame(matrix(ncol=length(list_treatments),nrow=1,
                             dimnames=list(c("Intercepts"), list_treatments)))
   
     for (t in list_treatments){
       df[1,t] <- input[[paste0("slider_treat_Int_", t)]]
     }
     df
   })
  
  #Visualize treatment parameters
  output$treatment_params_plot <- renderPlot({
    validate(
      need(input$slider_treat_k_0_Aggressiveness, label="Parameters")
    )
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    p_coeff <- treatment_params() %>%
      rownames_to_column(var="Variable") %>%
      pivot_longer(-Variable, names_to="Treatment", values_to="Parameter") %>%
      mutate(Variable=if_else(Variable=="Aggressiveness", "Agg.", Variable),
             Treatment=if_else(Treatment=="k_0", paste0('Y(0,', Treatment, ')'), paste0('Y(1,', Treatment, ')')) %>%
               factor(levels=c("Y(0,k_0)", paste0('Y(1,', input$list_treatments, ')')))) %>%
      ggplot(aes(x=Treatment, y=Variable, fill=Parameter)) +
      geom_tile(color="white") +
      geom_text(aes(label=Parameter)) +
      scale_fill_paletteer_c("pals::ocean.balance", limits=c(-50, 50)) +
      theme_pubclean() +
      theme(legend.position = "bottom") +
      labs(#title = "Linear regression coefficients",
           subtitle = "Linear regression coefficients",
           x = "Response to treatments",
           y = "Regr. coeff.")
    
     p_inter <- treatment_inter() %>%
       rownames_to_column(var="Variable") %>%
       pivot_longer(-Variable, names_to="Treatment", values_to="Parameter") %>%
       mutate(Treatment=if_else(Treatment=="k_0", paste0('Y(0,', Treatment, ')'), paste0('Y(1,', Treatment, ')')) %>%
                factor(levels=c("Y(0,k_0)", paste0('Y(1,', input$list_treatments, ')')))) %>%
       ggplot(aes(x=Treatment, y=Variable, fill=Parameter)) +
       geom_tile(color="white", show.legend = FALSE) +
       geom_text(aes(label=Parameter), show.legend = FALSE) +
       scale_fill_paletteer_c("pals::ocean.balance", limits=c(-50, 50)) +
       theme_pubclean() +
       theme(axis.text.y = element_blank(),
             axis.ticks.y = element_blank(),
             axis.title.x = element_blank()) +
       labs(subtitle = "Linear regression intercepts",
            title = paste0("Y(a, k_a) ~ ", paste0(list_genes, collapse = ' + ')),
            #x = "Response to treatments",
            y = "Intercepts")
    
    (p_inter / p_coeff) +
      plot_layout(heights = c(1,3))

  })
  
  
  #SIMULATE DATA WITH LAVA
  simulated_model <- eventReactive(input$start_simulation, {
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments_all <- c("k_0",input$list_treatments)
    genes_prevalence <- input$genes_prevalence %>% hot_to_r
    
    m <- lvm()
    
    #Define nature of variables
    #Agressiveness: corresponds to confounding factors of response. Indeed, in PDX you can observe a correlation between response when treated and response when untreated, meaning that some tumours are "intrinsically" more aggressive than others whatever the treatment status
    latent(m) <- ~Aggressiveness
    
    #C_i (or Genes): Binary mutational status for all genes. Same prevalence. Potential overlaps between mutations
    for (g in list_genes){
      prev <- genes_prevalence[genes_prevalence$Genes==g, "Prevalence"]
      distribution(m, as.formula(paste0("~", g))) <- binomial.lvm(p=prev)
    }
    
    #Y_i: Reponse when treated by drug k_i. Depends on Aggressiveness and C_i
    for (t in list_treatments_all){
      covariance(m, as.formula(paste0("~Y_", t))) <- input[[paste0("slider_treat_Cov_",t)]]
      intercept(m, as.formula(paste0("~Y_", t))) <- input[[paste0("slider_treat_Int_",t)]]
      regression(m, as.formula(paste0("Y_", t, "~ Aggressiveness" ))) <- input[[paste0("slider_treat_", t, "_Aggressiveness")]]
      for (g in list_genes){
        regression(m, as.formula(paste0("Y_", t, "~ ", g))) <- input[[paste0("slider_treat_", t, "_", g)]]
      }
    }
    m
  })
  
  simulated_data <- eventReactive(input$start_simulation, {
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments_all <- c("k_0", input$list_treatments)
    set.seed(1)
    sim_data <- lava::sim(simulated_model(), input$slider_size_super_pop) %>%
      left_join(treatment_assignment_hot_reprocess() %>% select(-Probability),
                by=list_genes) %>%
      mutate(Rand = runif(nrow(.)))
    
    probs <- select(sim_data, starts_with("P_")) %>% as.matrix
    cumul <- probs %*% upper.tri(diag(ncol(probs)), diag = TRUE) / rowSums(probs)
    
    sim_data %<>% mutate(K=list_treatments_all[rowSums(Rand > cumul) + 1L],
                         A=if_else(K=="k_0", 0, 1)) %>%
      mutate(Y=mapply(function(x,y) if (is.na(y)) {NA_real_}
                      else {
                        .[x, paste0("Y_", y)]
                        },
                      1:nrow(.), .$K) %>% unlist,
             Y_PM=mapply(function(x,y) if (is.na(y)) {NA_real_}
                         else {
                           .[x,paste0("Y_", y)]
                           }, 1:nrow(.), .$K_PM) %>% unlist,
             Y_Random=select(., one_of(paste0("Y_", input$list_treatments))) %>% rowMeans,
             PM=if_else(K_PM==K, 1, 0)) %>%
      rownames_to_column(var="PATIENT_ID") %>%
      select(-Rand, -one_of(paste0("P_", list_treatments_all)))
      
  })
  
  #Simulated data plot output select
  output$simulated_data_select <- renderUI({
    selectizeInput(
      'simulated_data_select', 'Select the response you want to plot',
      choices = paste0("Y_", c(input$list_treatments, "k_0", "PM", "Random")),
      select="Y_PM"
    )
  })
  
  #Render plot
  output$simulated_lava_plot <- renderPlot({
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments_all <- c(input$list_treatments, "k_0")
    
    #Start data processing
    sim_data <- simulated_data() %>%
      filter(!is.na(K_PM))
    sim_data$set = apply(sim_data %>% select(one_of(list_genes)), 1,function(x) {
      paste(sort(names(x)[which(x == 1)]), collapse="-")
    })
    # Add a "None" genre
    sim_data$None=ifelse(sim_data$set=="", 1, 0)
    sim_data$set[sim_data$set==""] = "None"
    
    # Get order of genre groupings 
    set_order <- sim_data %>% group_by(set, K_PM) %>% 
      summarise(n = n()) %>%
      arrange(K_PM) %>% ungroup %>%
      mutate(set = factor(set, levels=set),
             prop=round(n/sum(n), digits=3),
             pos=NA)
    
    set_order$pos[1] <- set_order$prop[1]/2
    for(i in 2:nrow(set_order)){
      set_order$pos[i] <- sum(set_order$prop[1:(i-1)])+set_order$prop[i]/2
    }
    
    # Set order of genre groupings
    sim_data = sim_data %>% 
      mutate(set = factor(set, levels=set_order$set))
    
    #Left tile plot
    y_axis_figures <- c(seq(from=0, to=sum(set_order$n), by=2000), sum(set_order$n))/sum(set_order$n)
    y_axis_labels <- c(seq(from=0, to=sum(set_order$n), by=2000), sum(set_order$n)) %>% as.character
    
    tile_categories <- sim_data %>% 
      pivot_longer(cols=one_of(list_genes), names_to="key", values_to="value") %>%
      group_by(set, key, value) %>% 
      slice(1) %>% 
      ungroup %>% 
      mutate(key = factor(key),
             label = if_else(value==1, as.character(key), "")) %>% 
      left_join(select(set_order, -K_PM), by="set") %>%
      ggplot(aes(x=key, y=pos, height=prop)) +
      geom_tile(aes(fill=K_PM, alpha=as.factor(value)), color="white")  +
      geom_text(aes(label=label), colour="white", size=3,fontface = "bold") +
      theme_pubclean() +
      theme(plot.title = element_text(hjust = 0.5),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank()) +
      scale_fill_manual(values=data_colors_drugs()) +
      scale_alpha_manual(values = c(0.2,1)) +
      scale_y_continuous(breaks = y_axis_figures, label = y_axis_labels) +
      labs(x= "Genes",
           y = "Number of patients and combinations of mutations") +
      guides(fill=FALSE, alpha=FALSE)
    
    plot_BP <- left_join(sim_data, select(set_order, set, pos, n), by="set") %>%
      ggplot(aes_string(x=input$simulated_data_select, y="pos", group="pos")) +
      geom_boxploth(aes(fill=as.factor(K_PM), weight=sqrt(n)),
                    show.legend = F, varwidth = T, width=0.1) +
      scale_fill_manual(values = data_colors_drugs()) +
      ylim(c(0,1)) +
      theme_classic() + 
      theme(axis.title.y=element_blank(),
            axis.text.y=element_blank(),
            axis.line.y=element_blank(),
            axis.ticks.y=element_blank(),
            plot.margin=margin(r=-2),
            plot.title = element_text(hjust = 0.5))
    
    #Sankey plot with ggalluvial
    data_sankey <- mutate(sim_data,
                          set=factor(set, levels = set_order$set) %>% fct_rev,
                          K_PM=paste0("K_PM=", as.character(K_PM)) %>%
                            factor(levels=paste0("K_PM=",list_treatments_all)) %>%
                            droplevels %>% fct_rev,
                          K=paste0("K=", as.character(K)) %>%
                            factor(levels=paste0("K=",list_treatments_all)) %>%
                            droplevels %>% fct_rev) %>%
      group_by(set, K_PM, K) %>%
      summarise(Freq=n())
      
    colors_drugs_sankey <- data_colors_drugs()
    names(colors_drugs_sankey) <- paste0("K_PM=", names(colors_drugs_sankey))
    
    plot_sankey <- data_sankey %>%
      ggplot(aes(y = Freq, axis1 = set, axis2 = K_PM, axis3= K)) +
      geom_alluvium(aes(fill = K_PM),width = 1/3) +
      geom_stratum() +
      geom_text(stat = "stratum", label = c(paste0(set_order$prop*100,"%"),
                                            levels(data_sankey$K_PM %>% fct_rev),
                                            levels(data_sankey$K %>% fct_rev)),
                size=3) +
      scale_x_discrete(limits = c("% patients", "K_PM", "K"), expand = c(.01, .01)) +
      scale_fill_manual(values = colors_drugs_sankey) +
      theme_minimal() +
      theme(legend.position = "right",
            axis.title.y=element_blank(),
            axis.text.y=element_blank(),
            axis.line.y=element_blank(),
            axis.ticks.y=element_blank(),
            panel.grid = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.background = element_blank())
    
    
    ggarrange(tile_categories, plot_BP, plot_sankey,
              ncol=3, widths=c(2,3,3),
              top="Mutations, sensitivities and drug assignment in PM")
    
  })
  
  #Render table
  output$sim_table <- renderDataTable({
    simulated_data() %>%
      mutate_if(is.numeric, function(x) round(x, digits=2)) %>%
      rename(`Agg.`=Aggressiveness) %>%
      datatable(rownames=FALSE)
  })
  
  #Function for true effects
  true_effects <- function(df){
    df %<>% filter(!is.na(Y_PM)) %>%
      mutate(CE1=Y_PM-Y_k_0,
             CE2=Y_PM-Y,
             CE3=Y_PM-Y_Random)
    
    res <- data.frame(
      CE1=mean(df$CE1),
      CE2=mean(filter(df, A==1)$CE2),
      CE3=mean(df$CE3)
    )
    
    return(res)
  }
  
  #Compute estimates
  estimates <- eventReactive(input$start_estimates, {
    
    super_populations <- simulated_data()
    number_cohorts <- input$slider_number_sub_coh
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments <- input$list_treatments
    
    #Define output object
    res <- data.frame(Cohort=1:number_cohorts,
                      CE1_True=NA_real_, CE1_Naive=NA_real_,
                      CE1_Std=NA_real_, CE1_IPW=NA_real_, CE1_TMLE=NA_real_,
                      CE2_True=NA_real_, CE2_Naive=NA_real_,
                      CE2_Std=NA_real_, CE2_IPW=NA_real_, CE2_TMLE=NA_real_,
                      CE3_True=NA_real_, CE3_Naive=NA_real_,
                      CE3_Std=NA_real_, CE3_IPW=NA_real_, CE3_TMLE=NA_real_
    )
    
    for (g in list_genes){
      res[,paste0("Prop_CE", 1:3, "_", g)] <- NA_real_
      res[,paste0("Diff_CE", 1:3, "_", g)] <- NA_real_
    }
    for (t in list_treatments){
      res[,paste0("Prop_CE", 1:3, "_", t)] <- NA_real_
      res[,paste0("Diff_CE", 1:3, "_", t)] <- NA_real_
    }
    
    #COHORTS: define the sampling sub-cohorts
    set.seed(1)
    sample_patients <- replicate(number_cohorts, sample(1:nrow(super_populations), input$slider_size_sub_coh, replace = F), simplify = T)
    
    #LOOP: compute all subcohorts causal effects and compositions
    withProgress(message = 'Computing causal estimates', value = 0,{
      for (i in 1:number_cohorts){
        
        incProgress(1/number_cohorts, detail = paste0("Patient ", i))
        
        cohort_i <- filter(super_populations, PATIENT_ID %in% sample_patients[,i]) %>%
          filter(!is.na(Y_PM))
        
        #SUBCOHORT COMPOSITION2: Compute compositions (in C_i and k_i) of subcohorts o later analyse potential biases
        #First for CE1 cohorts
        res[i, paste0("Prop_CE1_", list_genes)] <- filter(cohort_i, PM==1 | A==0) %>% select(one_of(list_genes)) %>% colMeans
        res[i, paste0("Prop_CE1_", list_treatments)] <- filter(cohort_i, PM==1 | A==0) %>% group_by(K_PM, .drop=FALSE) %>%
          summarise(prop=n()/nrow(.)) %>% .$prop
        res[i, paste0("Diff_CE1_", list_genes)] <- (filter(cohort_i, PM==1) %>% select(one_of(list_genes)) %>% colMeans) - (filter(cohort_i, A==0) %>% select(one_of(list_genes)) %>% colMeans)
        res[i, paste0("Diff_CE1_", list_treatments)] <- (filter(cohort_i, PM==1) %>% group_by(K_PM, .drop=FALSE) %>% summarise(prop=n()/nrow(.)) %>% .$prop) -
          (filter(cohort_i, A==0) %>% group_by(K_PM, .drop=FALSE) %>% summarise(prop=n()/nrow(.)) %>% .$prop)
        
        #Then for CE2 cohorts
        res[i, paste0("Prop_CE2_", list_genes)] <- filter(cohort_i, PM==1 | A==1) %>% select(one_of(list_genes)) %>% colMeans
        res[i, paste0("Prop_CE2_", list_treatments)] <- filter(cohort_i, PM==1 | A==1) %>% group_by(K_PM, .drop=FALSE) %>%
          summarise(prop=n()/nrow(.)) %>% .$prop
        res[i, paste0("Diff_CE2_", list_genes)] <- (filter(cohort_i, PM==1) %>% select(one_of(list_genes)) %>% colMeans) - (filter(cohort_i, A==1) %>% select(one_of(list_genes)) %>% colMeans)
        res[i, paste0("Diff_CE2_", list_treatments)] <- (filter(cohort_i, PM==1) %>% group_by(K_PM, .drop=FALSE) %>% summarise(prop=n()/nrow(.)) %>% .$prop) -
          (filter(cohort_i, A==1) %>% group_by(K_PM, .drop=FALSE) %>% summarise(prop=n()/nrow(.)) %>% .$prop)
        
        #Finally for CE3 cohorts
        res[i, paste0("Prop_CE3_", list_genes)] <- res[i, paste0("Prop_CE2_", list_genes)]
        res[i, paste0("Prop_CE3_", list_treatments)] <- res[i, paste0("Prop_CE2_", list_treatments)]
        res[i, paste0("Diff_CE3_", list_genes)] <-res[i, paste0("Diff_CE2_", list_genes)]
        res[i, paste0("Diff_CE3_", list_treatments)] <- res[i, paste0("Diff_CE2_", list_treatments)]
        
        #TRUE EFFECTS: Compute the true effects in the subcohort using counterfactual data
        res[i, paste0("CE", 1:3, "_True")] <- true_effects(cohort_i)
        
        #NAIVE EFFECTS: Compute the naive effects in the subcohort using only observed data
        res[i, "CE1_Naive"] <- mean(filter(cohort_i, PM==1)$Y) - mean(filter(cohort_i, A==0)$Y)
        res[i, "CE2_Naive"] <- mean(filter(cohort_i, PM==1)$Y) - mean(filter(cohort_i, A==1)$Y)
        res[i, "CE3_Naive"] <- mean(filter(cohort_i, PM==1)$Y) -
          filter(cohort_i, A==1) %>% group_by(K) %>% summarise(Mean=mean(Y)) %>% .$Mean %>% mean
        
        #COUNTERFACTUALS: estimate the counterfactual variables for later use in quantification of causal effects
        #Y_1_KPM: estimate the counterfactual outcome corresponding to all patients treated according to PM algorithm (used in CE1, CE2 and CE3)
        cohort_i_std <- mutate(cohort_i, K_1KPM=K)
        std1 <- glm(as.formula(paste0("Y ~ K_1KPM*(",
                                      paste0(list_genes, collapse = "+"),
                                      ")")),
                    data = cohort_i_std)
        cohort_i_std %<>% mutate(K_1KPM=K_PM) %>% mutate(Y_1KPM=predict(std1, .))
        
        #Y_1_k1, Y_1_k2 and Y_1_k3: estimate the counterfactual outcome corresponding to all patients treated with k1/k2/k3
        for (t in list_treatments){
          cohort_i_std[,paste0("Y_1_", t)] <- predict(std1, mutate(cohort_i_std, K_1KPM=t))
        }

        #Y_1_krand: estimate the counterfactual outcome corresponding to all patients treated with random treatment among k1/k2/k3 (for CE3)
        cohort_i_std[,"Y_1_krand"] <- cohort_i_std[, paste0("Y_1_", list_treatments)] %>% rowMeans

        #Y_0_k0: estimate the counterfactual outcome corresponding to all patients left untreated (for CE1)
        cohort_i_std %<>% mutate(A_01=A)
        std2 <- glm(as.formula(paste0("Y ~ A_01*(",
                                      paste0(list_genes, collapse = "+"),
                                      ")")),
                    data = cohort_i_std)
        cohort_i_std %<>% mutate(A_01=0) %>% mutate(Y_0_k0=predict(std2, .))
        #Y_1_K: estimate the counterfactual outcome corresponding to all patients treated with physician's treatment (for CE2)
        cohort_i_std %<>% mutate(A_01=1) %>% mutate(Y_1_K=predict(std2, .))
        
        #CAUSAL EFFECTS: estimate the causal effects
        res[i, "CE1_Std"] <- mean(cohort_i_std$Y_1KPM)-mean(cohort_i_std$Y_0_k0)
        res[i, "CE2_Std"] <- mean(cohort_i_std$Y_1KPM)-mean(cohort_i_std$Y_1_K)
        res[i, "CE3_Std"] <- mean(cohort_i_std$Y_1KPM)-mean(cohort_i_std$Y_1_krand)
        
        #IPW ESTIMATES: fit treatment models
        #Fit IPW models for CE1 and CE2
        #Weights for PM=1 arm
        cohort_i_ipw <- cohort_i
        
        #Weights for A=0 arm
        ipw_A_denom <- glm(as.formula(paste0("A ~ ",
                                             paste0(list_genes, collapse = "+"))),
                           data = cohort_i_ipw,
                           family = binomial(link = "logit")) %>%
          predict(type="response")
        ipw_A_nom <- mean(cohort_i_ipw$A)
        
        cohort_i_ipw %<>% mutate(SW_A=if_else(A==1, ipw_A_nom/ipw_A_denom, 
                                              (1-ipw_A_nom)/(1-ipw_A_denom)),
                                 W_A=if_else(A==1, 1/ipw_A_denom, 1/(1-ipw_A_denom)))
        
        #Fit IPW models for CE3
        ipw_K_denom <- multinom(as.formula(paste0("K ~ ",
                                                  paste0(list_genes, collapse = "+"))),
                                data = cohort_i_ipw, trace=FALSE) %>%
          predict(type="probs")
        ipw_K_num <- multinom(K~1,
                              data = cohort_i_ipw, trace=FALSE) %>%
          predict(type="probs")
        
        for (t in list_treatments){
          cohort_i_ipw %<>% mutate(UQ(paste0("SW_", t)) := ipw_K_num[,t]/ipw_K_denom[,t],
                                   UQ(paste0("W_", t)) := 1/ipw_K_denom[,t])
        }
        Y_IPW_rand <- sapply(list_treatments,
                             function(x) mean((cohort_i_ipw$K==x)*cohort_i_ipw[,paste0("SW_", x)]*cohort_i_ipw$Y) /
                               mean((cohort_i_ipw$K==x)*cohort_i_ipw[,paste0("SW_", x)]))
        
        #Test for versions
        cohort_i_ipw %<>% mutate(SW_PM=mapply(function(x,y) if (is.na(y)) {NA_real_}
                                              else {
                                                .[x, paste0("SW_", y)]
                                              },
                                              1:nrow(.), .$K_PM) %>% unlist,
                                 W_PM=mapply(function(x,y) if (is.na(y)) {NA_real_}
                                             else {
                                               .[x, paste0("W_", y)]
                                             },
                                             1:nrow(.), .$K_PM) %>% unlist)
        
        #Write causal effects
        t_PM <- mean(with(cohort_i_ipw, PM*SW_PM*Y)) / mean(with(cohort_i_ipw, PM*SW_PM))
        res[i, "CE1_IPW"] <- t_PM - (mean(with(cohort_i_ipw, (1-A)*SW_A*Y)) / mean(with(cohort_i_ipw, (1-A)*SW_A)))
        res[i, "CE2_IPW"] <- t_PM - (mean(with(cohort_i_ipw, A*SW_A*Y)) / mean(with(cohort_i_ipw, A*SW_A)))
        res[i, "CE3_IPW"] <- t_PM - mean(Y_IPW_rand)
        
        #TMLE ESTIMATE WITH GLM
        #First compute the Q0 model based on K
        cohort_i_tmle <- mutate(cohort_i, K_1KPM=K)
        
        tl_glm1 <- glm(as.formula(paste0("Y ~ K_1KPM*(",
                                         paste0(list_genes, collapse = "+"),
                                         ")")),
                       data = cohort_i_tmle#, %>% filter(K_1KPM!="LEE011"),family = fam
                       )
        
        cohort_i_tmle %<>% mutate(K_1KPM=K_PM) %>%
          mutate(Q0_KPM=predict.glm(tl_glm1, ., type="response"),
                 Q0_K=tl_glm1$linear.predictors)
        
        for (t in list_treatments){
          cohort_i_tmle[,paste0("Q0_K", t)] <- predict.glm(tl_glm1, mutate(cohort_i_tmle, K_1KPM=t), type="response")
        }
        
        #And based on A
        cohort_i_tmle %<>% mutate(A_01=A)
        tl_glm2 <- glm(as.formula(paste0("Y ~ A_01*(",
                                         paste0(list_genes, collapse = "+"),
                                         ")")),
                       data = cohort_i_tmle#,family = fam
                       )
        
        cohort_i_tmle %<>% mutate(A_01=0) %>%
          mutate(Q0_A0=predict.glm(tl_glm2, ., type="response"),
                 Q0_A=tl_glm2$linear.predictors) %>%
          mutate(A_01=1) %>%
          mutate(Q0_A1=predict.glm(tl_glm2, ., type="response"))
        
        #Compute propensity scores
        prop_A <- glm(as.formula(paste0("A ~ ",
                                        paste0(list_genes, collapse = "+"))),
                      data = cohort_i_tmle,
                      family = binomial(link = "logit")) %>%
          predict(type="response") %>% unname
        cohort_i_tmle %<>% mutate(gA=if_else(A==1, prop_A, 1-prop_A))
        
        prop_K <- multinom(as.formula(paste0("K ~ ",
                                             paste0(list_genes, collapse = "+"))),
                           data = cohort_i_tmle, trace=FALSE) %>%
          predict(type="probs")
        
        for (t in list_treatments){
          cohort_i_tmle %<>% mutate(UQ(paste0("gK_", t)) := unname(prop_K[,t]))
        }
        
        cohort_i_tmle %<>% mutate(gKPM=mapply(function(x,y) if (is.na(y)) {NA_real_}
                                              else {
                                                unname(.[x, paste0("gK_", y)])
                                              },
                                              1:nrow(.), .$K_PM) %>% unlist)
        
        #CE1 estimation
        cohort_i_tmle_CE1 <- mutate(cohort_i_tmle,
                                    CE1=if_else(A==0, 0, if_else(PM==1, 1, NA_real_))) %>%
          filter(!is.na(CE1)) %>%
          mutate(H1=if_else(CE1==1, 1/gKPM, 0),
                 H0=if_else(CE1==0, 1/(1-gA), 0))
        
          fit <- glm(Y ~ -1 + H0 + H1 + offset(Q0_K),
                     data = cohort_i_tmle_CE1#,family = fam
                     ) %>%
            coef
          
          cohort_i_tmle_CE1 %<>% mutate(Q1_A0=Q0_A0+fit[1]*H0,
                                        Q1_KPM=Q0_KPM+fit[2]*H1)
        
        res[i, "CE1_TMLE"] <- with(cohort_i_tmle_CE1, mean(Q1_KPM-Q1_A0))
        
        #CE2 estimation
        cohort_i_tmle_CE2 <- bind_rows(filter(cohort_i_tmle, PM==1) %>% mutate(CE2=1),
                                       filter(cohort_i_tmle, A==1) %>% mutate(CE2=0)) %>%
          mutate(H1=if_else(CE2==1, 1/gKPM, 0),
                 H0=if_else(CE2==0, 1/(gA), 0))
        
          fit <- glm(Y ~ -1 + H0 + H1 + offset(Q0_K),
                     data = cohort_i_tmle_CE2#,family = fam
                     ) %>%
            coef
          
          cohort_i_tmle_CE2 %<>% mutate(Q1_A1=Q0_A1+fit[1]*H0,
                                        Q1_KPM=Q0_KPM+fit[2]*H1)
        
        res[i, "CE2_TMLE"] <- with(cohort_i_tmle_CE2, mean(Q1_KPM-Q1_A1))
        
        #CE3 estimation
        cohort_i_tmle_CE3 <- bind_rows(filter(cohort_i_tmle, PM==1) %>% mutate(CE3=1),
                                       filter(cohort_i_tmle, A==1) %>% mutate(CE3=0)) %>%
          mutate(H1=if_else(CE3==1, 1/gKPM, 0))
        
        for (t in list_treatments){
          cohort_i_tmle_CE3[,paste0("H0_", t)] <- if_else(cohort_i_tmle_CE3$K==t, 1/cohort_i_tmle_CE3[,paste0("gK_", t)], 0)
          
            fit <- glm(as.formula(paste0("Y ~ -1 + H0_", t," + H1 + offset(Q0_K)")),
                       data = cohort_i_tmle_CE3#,family = fam
                       ) %>%
              coef
            
            cohort_i_tmle_CE3[,paste0("Q1_K", t)] <-
              cohort_i_tmle_CE3[,paste0("Q0_K", t)]+fit[1]*cohort_i_tmle_CE3[,paste0("H0_", t)]
            cohort_i_tmle_CE3[,paste0("Q1_KPM", t)] <-
              cohort_i_tmle_CE3[,"Q0_KPM"]+fit[2]*cohort_i_tmle_CE3[,"H1"]
          
          cohort_i_tmle_CE3[, paste0("CE3_", t)] <- cohort_i_tmle_CE3[, paste0("Q1_KPM", t)]-
            cohort_i_tmle_CE3[, paste0("Q1_K", t)]
          
        }
        
        res[i, "CE3_TMLE"] <- colMeans(select(cohort_i_tmle_CE3, starts_with("CE3_"))) %>% mean
        
      }
    })
    
    
    res
  })
  
  #Estimates plot
  output$estimates_plot <- renderPlot({
    input_data <- estimates()
    CEi <- input$estimates_select
    te <- true_effects(simulated_data())
    dev_metric <- input$estimates_perf
    val <- gsub("CE", "", CEi)
    
    te_CEi <- te[CEi] %>% unlist %>% unname
    
    plot_data <- select(input_data, Cohort, starts_with(CEi)) %>%
      rename_all(funs(str_replace(., paste0(CEi,"_"), ""))) %>%
      mutate(TE=te_CEi)
    
    #Distributions of CE1 estimates
    p_distrib <-  pivot_longer(plot_data, any_of(c("True", "Naive", "Std", "IPW", "TMLE")),
                               names_to = "Method", values_to = "Estimate") %>%
      mutate(Method=factor(Method,
                           levels=c("True", "TMLE", "Std", "IPW", "Naive"))) %>%
      ggplot(aes(x=Method, y=Estimate, fill=Method)) +
      geom_boxplot(alpha=1, show.legend = FALSE, varwidth = FALSE, width=0.4) +
      geom_text(data = . %>% filter(!is.nan(Estimate)) %>% group_by(Method) %>%
                  summarise(N=paste0("n=",n()), Min=min(Estimate), Max=max(Estimate)) %>%
                  mutate(Estimate=min(Min)-0.05*(max(Max)-min(Min))),
                aes(x=Method, label=N), size=3) +
      scale_fill_manual(values=colors_methods) +
      labs(title=bquote(CE[.(val)] ~ " distributions")) +
      theme_pubclean() +
      theme(legend.position = "bottom",
            legend.title = element_text(face="bold"))
    
    #Deviations
    p_diff <-pivot_longer(plot_data, any_of(c("Naive", "Std", "IPW", "TMLE")),
                          names_to = "Method", values_to = "Estimate") %>%
      mutate(Method=factor(Method, levels=c("TMLE", "Std", "IPW", "Naive")),
             Diff=Estimate-True) %>%
      ggplot(aes(x=Method, y=Diff, fill=Method)) +
      geom_boxplot(alpha=1, show.legend = FALSE, varwidth = FALSE, width=0.4) +
      scale_fill_manual(values=colors_methods) +
      labs(title=bquote(CE[.(val)] ~ " deviations"),
           y="Deviation value") +
      theme_pubclean() +
      theme(legend.position = "bottom",
            legend.title = element_text(face="bold"))
    
    #Performances
    p_perf_plot <-pivot_longer(plot_data, any_of(c("Naive", "Std", "IPW", "TMLE")),
                          names_to = "Method", values_to = "Estimate") %>%
      mutate(Method=factor(Method, levels=c("TMLE", "Std", "IPW", "Naive"))) %>%
      group_by(Method) %>%
      summarise(RMSE=RMSE(True, Estimate, na.rm=TRUE), MAE=MAE(True, Estimate, na.rm=TRUE))
    
    if (dev_metric=="MAE"){
      p_perf <- mutate(p_perf_plot, Label=as.character(round(MAE, digits=2))) %>%
        ggplot(aes(x=Method, y=MAE, fill=Method)) +
        geom_bar(stat = "identity", width=0.5) +
        geom_text(aes(label=Label, y=0.5*min(MAE)), color="white") +
        scale_fill_manual(values=colors_methods,guide = guide_legend(direction = "vertical")) +
        theme_pubclean() +
        theme(legend.justification = "center",
              legend.text = element_text(size=8),
              legend.title = element_text(face="bold", size=10))
    } else {
      p_perf <- mutate(p_perf_plot, Label=as.character(round(RMSE, digits=2))) %>%
        ggplot(aes(x=Method, y=RMSE, fill=Method)) +
        geom_bar(stat = "identity", width=0.5) +
        geom_text(aes(label=Label, y=0.5*min(RMSE)), color="white") +
        scale_fill_manual(values=colors_methods,guide = guide_legend(direction = "vertical")) +
        theme_pubclean() +
        theme(legend.justification = "center",
              legend.text = element_text(size=8),
              legend.title = element_text(face="bold", size=10))
    }
    
    
    #Scatter plot for estimates
    p_scatter <- pivot_longer(plot_data, any_of(c("Naive", "Std", "IPW", "TMLE")),
                              names_to = "Method", values_to = "Estimate") %>%
      mutate(Method=factor(Method, levels=c("TMLE", "Std","IPW", "Naive"))) %>%
      ggplot(aes(x=True, y=Estimate)) +
      geom_point(aes(color=Method), size=0.6, show.legend = FALSE) +
      geom_abline(aes(slope = 1, intercept = 0, linetype = "y = x"),
                  lwd=1,show.legend=TRUE) +
      geom_smooth(method = "lm", formula =y ~ x, aes(linetype = "y = ax+b"),
                  color="black",show.legend=FALSE) +
      stat_cor(aes(label = `..r.label..`), p.digits =1, label.sep = '\n') +
      facet_grid(.~Method)  +
      scale_color_manual(values=c(colors_methods)) +
      scale_linetype_manual(values= c("y = x"="solid", "y = ax+b"="dashed"),
                            guide = guide_legend(direction = "vertical",
                                                 override.aes=list(fill=NA))) +
      labs(title = bquote(CE[.(val)] ~ " estimates compared to true effects"),
           x="True estimates",
           y="Estimates from\nobserved data",
           linetype="Lines") +
      theme_pubclean() +
      theme(legend.justification = "center",
            legend.text = element_text(size=8),
            legend.title = element_text(face="bold", size=10),
            legend.key = element_rect(fill = NA, colour = NA)) +
      guides(color=FALSE)
    
    ((p_distrib / (p_diff + p_perf) / p_scatter) | guide_area()) +
      plot_layout(guides = 'collect', widths = c(6,1))

  })
  
  output$estimates_plot_PCA <- renderPlot({
    input_data <- estimates()
    list_genes <- input$genes_prevalence %>% hot_to_r %>% .$Genes
    list_treatments <- input$list_treatments
    CEi <- input$estimates_select
    val <- gsub("CE", "", CEi)
    PCA_var <-input$estimates_PCA_var
    PCA_enrich <- input$estimates_PCA_enrich
    PCA_color <- input$estimates_PCA_color
    
    plot_data_pca <-  select(input_data, Cohort, starts_with(paste0(c("", "Prop_", "Diff_"), CEi))) %>%
      rename_all(funs(str_replace(., paste0(CEi,"_"), ""))) %>%
      mutate(Dev_Naive=(Naive-True), Dev_Std=(Std-True), Dev_IPW=(IPW-True), Dev_TMLE=(TMLE-True))
    
    if (PCA_var == "Genes"){
      pca_var <- list_genes
    } else if (PCA_var == "K_PM"){
      pca_var <- list_treatments
    } else if (PCA_var =="Both"){
      pca_var <- c(list_genes, list_treatments)
    }
    
    if (PCA_enrich == "Proportions"){
      pca_enrich <- "Prop"
    } else if (PCA_enrich == "Enrichment"){
      pca_enrich <- "Diff"
    } else if (PCA_enrich =="Both"){
      pca_enrich <- c("Prop", "Diff")
    }
    
    all_var <- cross(list(pca_enrich, pca_var)) %>% map_chr(paste, collapse = "_")
    
    pca_data <- PCA(X = select(plot_data_pca, one_of(all_var)), graph = FALSE, ncp = 2)
    pca_data_pcs <- pca_data$ind$coord[,1:2]
    plot_data_pca %<>% cbind.data.frame(pca_data_pcs)
    
    if (PCA_color == "Absolute CE"){
      plot_data_pca %<>% pivot_longer(any_of(c("Naive", "Std", "IPW", "TMLE")),
                                      names_to = "Method", values_to = "Estimate") %>%
        mutate(Method=factor(Method, levels=c("TMLE", "Std","IPW", "Naive")))
      l <- c(min(plot_data_pca$Estimate, na.rm = T), max(plot_data_pca$Estimate, na.rm = T))
    } else if (PCA_color == "Deviation from true CE"){
      plot_data_pca %<>% pivot_longer(any_of(c("Dev_Naive", "Dev_Std", "Dev_IPW", "Dev_TMLE")),
                                      names_to = "Method", values_to = "Estimate") %>%
        mutate(Method=factor(Method, levels=c("Dev_TMLE", "Dev_Std","Dev_IPW", "Dev_Naive")))
      l <- c(-max(abs(plot_data_pca$Estimate), na.rm = T), max(abs(plot_data_pca$Estimate), na.rm = T))
    } 
    
    #Compute PCA utilities for further plots
    axes <- c(1, 2)
    var <- facto_summarize(pca_data, element = "var", result = c("coord","contrib", "cos2"), axes = axes)
    pca_ind <- get_pca_ind(pca_data)
    ind <- data.frame(pca_ind$coord[, axes, drop = FALSE])
    r <- min((max(ind[, "Dim.1"]) - min(ind[, "Dim.1"])/(max(var[, "Dim.1"]) - 
                                                           min(var[, "Dim.1"]))), (max(ind[, "Dim.2"]) - min(ind[, "Dim.2"])/(max(var[, "Dim.2"]) - min(var[, "Dim.2"]))))
    
    p_arrow_fviz <- fviz_pca_var(pca_data, scale. = r, repel = T)
    data_arrow <- p_arrow_fviz$data %>% select(name, x, y)
    
    p_pca <- ggplot() +
      geom_point(data = plot_data_pca,
                 aes(x=Dim.1, y=Dim.2,color=Estimate)
      ) +
      scale_color_paletteer_c("pals::ocean.delta", limits=l) +
      geom_segment(data=data_arrow,
                   aes(x=0, y=0, xend=x, yend=y),
                   arrow = arrow(angle = 30, length = unit(2, "mm"),
                                 ends = "last", type = "open")) +
      geom_text_repel(data=data_arrow,
                      aes(label=name, x=1.05*x, y=1.05*y),
                      box.padding = 0.25, point.padding = 1e-06,
                      segment.size = 0.2, min.segment.length = 0.5) +
      facet_grid(.~Method) +
      labs(title = bquote("Dependance of " ~ CE[.(val)] ~ " to confounders"),
           x=p_arrow_fviz$labels$x,
           y=p_arrow_fviz$labels$y,
           color=PCA_color) +
      theme_pubclean() +
      theme(legend.position = "bottom",
            legend.title = element_text(face="bold"))
    
    p_pca
    
  })
  
  #Estimates table
  output$estimates_table <- renderDataTable({
    estimates() %>%
      mutate_if(is.numeric, function(x) round(x, digits=2)) %>%
      datatable(rownames=FALSE)
  })
  
  #Target trials pictures
  output$photo <- renderImage({
    list(
      src = file.path("Pictures/Target_Trials.png"),
      width =600
    )
  }, deleteFile = FALSE)

}

# Run the application 
shinyApp(ui = ui, server = server)

