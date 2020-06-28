# Causal Precision Medicine

This repository contains all the code and data required to reproduce the article "Causal inference with multiple versions of treatment and application to personalized medicine"

There is one .Rmd file for the analyses on simlated data and the corresponding plots used in the article (Analysis\_Sim.Rmd). The document compiles in about 5/6 minutes and the resulting .html document is provided (Analysis\_Sim.html). 
This static analysis is complemented by an interactive RShiny app (Application\_Causal\_PM) to allow straightforward tests of user-defined simulation scenarios. This application is also available online (https://jonasbeal.shinyapps.io/application_causal_pm/). However, its use is limited in terms of number of active hours per month. Please use the .R file rather than the online version if you are able to do so.

There is another .Rmd file for the analyses on PDX data which are otherwise provided. Data come from Gao et al. (https://doi.org/10.1038/nm.3954). The document compiles in about 2 hours and the resulting .html document is provided (Analysis\_PDX.html). 
