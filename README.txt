This is an instruction to reproduce the work done in “Machine Learning Enhanced Urban Analysis for Forecasting Morphogenesis and Housing Allocation in Rapidly Developing Cities.”

1- The images folder contains the preprocessed binary maps, as described in the manuscript. The original dataset can be downloaded from:
https://human‑settlement.emergency.copernicus.eu/download.php?ds=bu

2- To visualize these images, you can either use Python (e.g., Matplotlib) or GIS software such as QGIS or ArcGIS.

3- Run the main.py file to train the ML model. All hyperparameters are set to their final tuned values.

4- Run the predict.py file to generate the forecasted map. The first run will predict the next time step—for example, supplying 2010, 2015, and 2020 will produce the 2025 map. To predict 2030, use 2015, 2020, and the newly generated 2025 map, and repeat as needed.