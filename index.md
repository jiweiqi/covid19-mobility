---
layout: default
---

The Pandemic Oil Demand Analysis (PODA) model aims to project the motor gasoline demand for the medium-term—three months into the future—based on the evolution of the COVID-19 pandemic and the resulting changes in travel mobility. Through connecting the on-going changes in personal travel patterns and evolution of the COVID-19 pandemic, the PODA model offers an understanding of gasoline demand which can be beneficial to economic and transportation energy planning and policy discussions.


## US Motor Gasoline Demand Projection - Updated Weekly


<p align="center">
  {% include iframe.html %}
</p>


## Model structure

As the figure below shows, the PODA model contains two major modules. The first module, the Mobility Dynamic Index Forecast module, identifies the changes in travel mobility caused by the evolution of COVID-19 pandemic and government orders, and projects the changes in travel mobility indices relative to the normal level in the U.S. Notably, the change in travel mobility, which affects the frequency of human contact or the level of social distancing, can reciprocally impact the evolution of the pandemic to some extent, as the dashed line shows in the following figure. However, to simplify the model, we ignore the dynamic feedback in the current PODA model version. The second major PODA module, the Motor Gasoline Demand Estimation module, estimates vehicle miles traveled on pandemic days while considering the dynamic indices of travel mobility, and quantifies motor gasoline demands by coupling the gasoline demands and vehicle miles traveled. The COVID-19 pandemic projections in this study are supported by the [YYG model](https://covid19-projections.com), and the mobility-related information is based on the public data released by [Google](https://www.google.com/covid19/mobility/) and [Apple](https://www.apple.com/covid19/mobility).

![Image]({{ site.url }}/assets/PODA_Model.png)

## Acknowledgement

This study was financially supported by Aramco Services Company and used resources at the National Transportation Research Center at Oak Ridge National Laboratory and the Energy Systems Center at Argonne National Laboratory. The authors are solely responsible for the views expressed in this study.

## Contacts

Shiqi Ou <ous1@ornl.gov>, Xin He <xin.he@aramcoamericas.com>, Weiqi Ji <weiqiji@mit.edu>