# PODA Model User Manual

This directory hosts all of the python code and data/links for training the model and make projection.

## Dependence
* pytorch
* sklearn
* shap
* pandas
* seaborn
* tqdm
* numpy
* matplotlib

## Download Pandenmic and Mobility Data
The PODA Python codes need some adjustments to download pandemic and mobility data. 
1.  Save all files in the same folder
2.	Adjust the following variables in 
    
    a.	`PODA_1_Data_Processing_for_ML.py` to download data from website automatically:
       
       i.	`YYG_date_adjust = 1`. 
       
            (need to check [YYG projections website](https://github.com/youyanggu/covid19_projections/tree/master/projections) finding the latest update. `YYG_date_adjust` is the day difference between today and the latest YYG projection).
       
       ii.	`Apple_Date_adjust = 1`

            df_Apple_Mobility = pd.read_csv("https://covid19-static.cdn-apple.com/covid19-mobility-data/2009HotfixDev27/v3/en-us/applemobilitytrends-"+Apple_File_Date+".csv")
(Check [Apple website](https://www.apple.com/covid19/mobility). You need to find the hidden date and file under the button shown below, using “Developer Tools”. Update “Apple_Date_adjust “ and “df_Apple_Mobility”.
 
    b.  `PODA_3_MIT_Data_Processing_For_Projection.py`. 
        
        The MIT data needs to be manually downloaded and saved in the model folder. You need to adjust the Python code to match the file name you saved.
MIT_file_name = 'MIT_covid_analytics_projections-2020-06-17.csv' 

 3.	Run the following Python codes with the sequence of the number order:
    
    a.	PODA_1_Data_Processing_for_ML.py
    b.	PODA_2_Data_Processing_For_Projection.py
    c.	PODA_3_MIT_Data_Processing_For_Projection.py
    d.	PODA_4_ReLU_mobility_training.py
    e.	PODA_5_ML_Mobility Prediction.py
    f.	PODA_6_GoogleMobility_EIA_Correlation.py
    g.	PODA_7_Apple_EIA_Correlation.py
    h.	PODA_8_Fuel Demand Projection.py
    
    All results, including the data downloaded online will be saved in file ”PODA_Model_xxxx_xx_xx.npy”, in which “xxxx_xx_xx” represents the date (in format yyyy_mm_dd) of the model.

You can also find a detailed instruction on how to runing the code in the word document of `Readme.docx`.
