# Water potability prediction in Python

## Project explanation
The objective of this project is to analyze and study, through the use of supervised and unsupervised machine learning methods, the potability of water from data in the 'drinking water potability' dataset. The project is divided into 3 parts:
- **Exploratory Data Analysis (EDA)**. This is the phase in which an exploratory analysis of the entire dataset is carried out. Initially, the individual variables in the dataset will be analyzed, showing their main statistical metrics (mean, standard deviation, quartiles, minimum and maximum values, etc.), the presence or absence of missing values, and, if they are present, the consequent action to be taken (computation, replacement with mean/median, etc.). Then a multivariate analysis will be carried out, with the aim of looking for possible relationships (collinearity and/or multicollinearity) between the variables. \
The jupyter notebook related to this step can be found at the following: [link](https://github.com/opon13/Drinking-water-potability-analysis-in-Python/blob/main/Exaploratory%20Data%20Analysis%20(EDA).ipynb). \
At the end of the analysis, the results will be exposed and two scripts in python will be written from them, the purpose of which is to model (Data Scaling and Preprocessing) the dataset according to the results obtained (e.g., possible removal of variables and/or observations with a high rate of missing values, removal of variables with a high correlation index with others). \
The scripts inherent to Data Scaling and Preprocessing can be found at the following: [link]().
- **Model-building phase**. In this one will try to build a machine learning model that can classify/predict the output of the target variable 'Potability'. To do this, supervised and non-supervised learning techniques will be applied. \
The jupyter notebook related to model construction using supervised learning techniques can be found at the following link: [supervised](). \
The jupyter notebook related to building the model using the unsupervised learning techniques can be found at the following link: [unsupervised]().
- **Evaluation phase**. In this phase, the metrics obtained from the various models will be compared and the model that obtained the highest results will be chosen.

## The Dataset
You can view and download the dataset at the following: [link](https://www.kaggle.com/datasets/artimule/drinking-water-probability)

### Context
Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. This is important as a health and development issue at a national, regional, and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

### Content
The drinkingwaterpotability.csv file contains water quality metrics for 3276 different water bodies.

- pH value:
    PH is an important parameter in evaluating the acid-base balance of water. It is also the indicator of the acidic or alkaline condition of water status. WHO has recommended the maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

- Hardness:
    Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness-producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

- Solids (Total dissolved solids - TDS):
    Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates, etc. These minerals produced an unwanted taste and diluted color in the appearance of water. This is the important parameter for the use of water. The water with a high TDS value indicates that water is highly mineralized. The desirable limit for TDS is 500 mg/l and the maximum limit is 1000 mg/l which is prescribed for drinking purposes.

- Chloramines:
    Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

- Sulfate:
    Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

- Conductivity:
    Pure water is not a good conductor of electric current rather’s a good insulator. An increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceed 400 μS/cm.

- Organic_carbon:
    Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

- Trihalomethanes:
    THMs are chemicals that may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm are considered safe in drinking water.

- Turbidity:
    The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of the light-emitting properties of water and the test is used to indicate the quality of waste discharge with respect to the colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

- Potability:
    Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.

