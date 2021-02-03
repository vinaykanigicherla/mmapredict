# mmapredict
## Description 
Being an avid MMA fan, I've always wanted to be able to predict the outcome of a fight. This project is my little stab at it. The project utilizes trains and tunes a diverse set of classifiers which it then ensembles using Stacking. The meta-classifier used in the stacking model is also tuned. 

The raw data is obtained through scraping historical UFC fight data from ufcstats.com. This data is then processed to create a training dataset containing the two fighters, the winner of the bout, and the feature engineered stats for each fighter at the time of the fight (see fighterinfo_features.txt in the "data" directory for a list of all the computed features). 

(Note project is still under progress.)

## Usage 
1) Clone the repo:

```
$ git clone https://github.com/vinaykanigicherla/mmapredict.git
```

2) Run scrapy web crawler from "mma_scraper" directory, save data, and move the file to "data" directory:

```    
$ cd mma_scraper/
$ scrapy crawl fight_scraper -O raw_data.json
$ mv raw_data.json ../data/
```

3) Run data_processing script (steps visualized in Jupyter Notebooks in "data_processing" directory)

```    
$ python data_processing/data_processing.py
```

4) Train and save baseline models. Tune hyperparameters of baseline models and save. 

```    
$ python training/lvl1_models.py --data_path "data/train.pkl" --seed 0
```

5) Train and save stacking model. 
    
```
$ python training/stacking.py --data_path "data/train.pkl" --seed 0
```

## TODO
* Implement feature selection
* Investigate if perofrmance increases possible through training different lvl1 classifiers on different subsets of data
* Investigate scope for additional feature engineering

