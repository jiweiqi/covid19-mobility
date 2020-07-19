---
layout: default
---

During this COVID-19 pandemic, we have seen a significant drop in mobility and fuel demand. We take a data-driven approach to understand the impact of COVID-19 statistics and states policy on personal mobility. To give you a role of thumb, the mobility quickly dropped as the number of confirmed cases increasing in March. Along with the decreasing of confirmed cases and the ease of stay home order from May, the mobility has gradually recovered. We also build a model to connect the mobility changes to fuel demand changes.

We then take the projected COVID-19 infection case numbers from the epidemiology model as inputs to predict future trends of mobility and fuel demand. We hope those projections could benefit economic and transportation energy planning and policy discussions.

You can find more details from our paper in **Nature Energy** [Machine learning model to project the impact of COVID-19 on US motor gasoline demand](https://www.nature.com/articles/s41560-020-0662-1)

## US Motor Gasoline Demand Projection - Updated Weekly

Looking ahead, there are significant uncertainties in the COVID-19 projections and Fuel Demand projections depends on whether people will continue social distancing in the reopening phases. In the reference scenario, it is assumed that people will still largely keep social distancing and states will re-impose stay home order if there is a second wave. Therefore, we see a flatten projection curve. In the optimistic cases, people will keep social distancing even the stay-home-order is eased. In the pessimistic scenario, the fuel demand could face a second wave of droping due to the second wave of COVID-19 pandemic.

**Projection made on July 15**
<p align="center">
 {% include iframe.html %}
</p>
Noted that EIA data was shifted 7 day to represent the delay between actual fuel use, refueling, and EIA data reporting. We also include the COVID-19 projections from [YYG](https://covid19-projections.com) here for your reference.

<p align="center">
 {% include iframe_YYG_US.html %}
</p>

## Model Structure

Our model is called Pandemic Oil Demand Analysis (PODA) model. The PODA model aims to project the motor gasoline demand for the medium-term—three months into the future—based on the evolution of the COVID-19 pandemic and the resulting changes in travel mobility. Through connecting the on-going changes in personal travel patterns and evolution of the COVID-19 pandemic, the PODA model offers an understanding of gasoline demand which can be beneficial to economic and transportation energy planning and policy discussions.

As the figure below shows, the PODA model contains two major modules. The first module, the Mobility Dynamic Index Forecast module, identifies the changes in travel mobility caused by the evolution of COVID-19 pandemic and government orders and projects the changes in travel mobility indices relative to the normal level in the U.S. Notably, the change in travel mobility, which affects the frequency of human contact or the level of social distancing, can reciprocally impact the evolution of the pandemic to some extent, as the dashed line shows in the following figure. However, to simplify the model, we ignore the dynamic feedback in the current PODA model version. The second major PODA module, the Motor Gasoline Demand Estimation module, estimates vehicle miles traveled on pandemic days while considering the dynamic indices of travel mobility and quantifies motor gasoline demands by coupling the gasoline demands and vehicle miles traveled. The COVID-19 pandemic projections in this study are supported by the [YYG model](https://covid19-projections.com), and the mobility-related information is based on the public data released by [Google](https://www.google.com/covid19/mobility/) and [Apple](https://www.apple.com/covid19/mobility).

![Image]({{ site.url }}/assets/PODA_Model.png)


## Our Team

   * [Shiqi Ou](https://www.linkedin.com/in/shiqi-shawn-ou-9137a149/) (Energy and Transportation Science Division, Oak Ridge National Laboratory)
   * [Xin He](https://www.linkedin.com/in/xin-he-11035b14/) (Aramco Services Company) 
   * [Weiqi Ji](https://www.linkedin.com/in/weiqiji/) (Massachusetts Institute of Technology) 
   * Wei Chen (Michigan Department of Transportation) 
   * Lang Sui (Aramco Services Company) 
   * Yu Gan (Energy Systems Division, Argonne National Laboratory) 
   * Zifeng Lu (Energy Systems Division, Argonne National Laboratory) 
   * Zhenhong Lin (Energy and Transportation Science Division, Oak Ridge National Laboratory) 
   * [Sili Deng](https://deng.mit.edu/people.html) (Massachusetts Institute of Technology) 
   * Steven Przesmitzki (Aramco Services Company) 
   * Jessey Bouchard (Aramco Services Company)
