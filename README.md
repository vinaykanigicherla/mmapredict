# mmapredict
## Description 
Being an avid MMA fan, I've always wanted to be able to predict the outcome of a fight. This project is my little stab at it. The project utilizes trains and tunes a diverse set of classifiers which it then ensembles using Stacking. The meta-classifier used in the stacking model is also tuned. 

The raw data is obtained through scraping historical UFC fight data from ufcstats.com. This data is then processed to create a training dataset containing the two fighters, the winner of the bout, and the feature engineered stats for each fighter at the time of the fight (see fighterinfo_features.txt in data_processing/ for a list of all the features per fighter). 

## Usage 

