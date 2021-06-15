
# Introduction
The following code helps in creating models for predicting one day cricket matches, evaluate and inference
It sues combination of engineered features, learned embedding features for countries and learned embedding features for batsman
Prediction is done in two steps. First innings(regression) run and second  innings chase success (classification)


# setup PYTHONPATH in cricketai directory
export PYTHONPATH=.:$PYTHONPATH


# Process wise steps

-- for all commands use --help to know more

## Data Preparation

### download from cricinfo
python odi/data_loader/cricinfo_scraper.py load-match-cricinfo --year_list 2009 --year_list 2010 --append n

### clean data downloaded fromcricinfo data
python odi/data_loader/cricinfo_scraper.py clean-cricinfo

## Preprocessing - Creating Ranks (Depends on loaded data)
Create ranks of participating batsman, bowler and countries

### create all ranking for list of years
python odi/preprocessing/rank-cricinfo.py all --year_list 2014 --year_list 2015

python odi/preprocessing/rank-cricinfo.py all
(for current year only)

ranking is created for all mentioned years once every month for 1 yea r ending on the month end data

### create all ranking from current to 2 years previous
python odi/preprocessing/rank-cricinfo.py all --no_of_years 2
### create batsman ranking (will also create country)
python odi/preprocessing/rank-cricinfo.py.py batsman

python odi/preprocessing/rank-cricinfo.py.py bowler

python odi/preprocessing/rank-cricinfo.py.py location
### create only bowler ranking (without country)
python odi/preprocessing/rank-cricinfo.py.py bowler-only

python odi/preprocessing/rank-cricinfo.py.py bowler-only


## Inferencing
### verify team and location names
python odi/model_util/input_helper_ci.py find-location --location kolkata

### Create input template
python odi/model_util/input_helper_ci.py create-input-template --team_a India --team_b Australia --location Kolkata

(optionally might want to modify inference_config.json before inferencing and choose combination of mdoels with or without embedding)

### predict outcome
python odi/inference/prediction_ci.py match

-by default team_a.xlsx and team_b.xlsx will be used, type help the check options with --hel[]

-no_of_years is only applicable while considering trend calculation

- check the --help


### get optimum first innings run
python odi/inference/prediction_ci.py optimize

- by default team_a.xlsx is considered as team and team_b.xlsx is considered as opponent

- team is considered as batting first

- check options with --help



## Retraining with cricinfo (Necessary data loading and Ranking has been done)
### Step 1 - Select highly correlated score features (optional - only for viewing)
python odi/retrain/retrain_ci.py select-score-cols --start_date 2011-01-01 --end_date 2019-01-01

### Step 2 - Create train test for first innings base model
python odi/retrain/create_train_test_ci.py first-innings-base --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

--can optionally use sequential feature selection

### Step 3 - Create first innings base model 
python odi/retrain/retrain_ci.py first-innings-regression

- optional neural network training
python odi/retrain/retrain_ci.py train-first-innings-base-neural

### Step 4 - Create train test for second innings base model
python odi/retrain/create_train_test_ci.py second-innings-base --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

--can optionally use sequenctial feature selection

### Step 5 - Create second innings base model 
python odi/retrain/retrain_ci.py second-innings-classification

### Step 6 - create train test for second level training by making both teams as both innings without neural network
python odi/retrain/create_train_test_ci.py second-level-non-neural --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

### Step 7 - retrain classification to predict outcome with both teams as both innings without neural network
python odi/retrain/retrain_ci.py combined-non-neural

### step 8 create one shot train test
python odi/retrain/create_train_test_ci.py one-shot --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

### Step 9 - Create second innings base model 
python odi/retrain/retrain_ci.py one-shot-classification

### step 10 create one shot neural network with multi output train test
python odi/retrain/create_train_test_ci.py one-shot-multi --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

### step 11 train multi output neural in one shot
python odi/retrain/retrain_ci.py train-multi-output-neural --epoch 500

-- if it does not come to accruracy 70 restart the training or try --mode tune
-- both train and test can be around 70 with a couple of trials

### Step 12 - create train test for second level training by making both teams as both innings
python odi/retrain/create_train_test_ci.py second-level-any --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

### Step 13 - retrain classification to predict outcome with both teams as both innings
python odi/retrain/retrain_ci.py combined-any-innings


### Step 14 compare with previous paper - create traine test for mg
python odi/retrain/create_train_test_ci.py mg --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2020-12-31

### Step 15 - retrain mg(prior paper) classification to predict outcome
python odi/retrain/retrain_ci.py mg-classification









# DEPRECATED COMMANDS

## Loading matches
Load match details in local file system
### Current data (incremental)
python odi/data_loader/data_loader.py load-current
### For refreshing entire data
python odi/data_loader/data_loader.py load-current --append n
### For provinding inclusive date range
python odi/data_loader/data_loader.py load-current --from_date 2014-01-01 --to_date 2020-12-31
### for loading previous data
python odi/data_loader/data_loader.py load-old --from_date 2011-01-01 --to_date 2013-12-31
### load list of not batted
python odi/data_loader/cricinfo_scraper.py load-not-batted --year_list 2009 --year_list 2010 --append n

### update match stats for not batted-should be done after ranking
python odi/data_loader/cricinfo_scraper.py update-stats
### remove incorrect matches 
python odi/data_loader/cricinfo_scraper.py remove-incorrect


### create player score mean reduction factor by position
python odi/preprocessing/rank.py reduction-analysis --start_date 2016-01-01 --end_date 2019-01-01


### update match stats for not batted
python odi/data_loader/cricinfo_scraper.py update-stats
### remove incorrect mathces
python odi/data_loader/cricinfo_scraper.py remove-incorrect --start_date 2009-01-01

### individual run prediction
python odi/inference/prediction.py individual-runs






## Retraining (Necessary data loading and Ranking has been done)
### Step 8 - Create one hot encoding for batsman, location and country
python odi/retrain/create_encoding_ci.py location --start_date '2004-01-01' --end_date '2018-12-31'

python odi/retrain/create_encoding_ci.py country --start_date '2004-01-01' --end_date '2018-12-31'

### Step 9 - Create train-test files for learning country embedding
python odi/retrain/create_train_test_ci.py country-embedding --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2019-12-31

### Step 10 - Learn country embedding
python odi/retrain/retrain_ci.py train-country-embedding
#### tune country embedding
python odi/retrain/retrain_ci.py train-country-embedding --mode tune --epoch 10 --learning_rate 0.0001
#### check country embedding
python odi/retrain/retrain_ci.py check-country-embedding

### Step 11 - Create train-test files for learning country embedding
python odi/retrain/create_train_test_ci.py country-embedding-2nd --train_start 2004-01-01 --test_start 2019-01-01 --test_end 2019-12-31

### Step 12 - Learn country embedding
python odi/retrain/retrain_ci.py train-country-embedding-2nd
#### tune country embedding
python odi/retrain/retrain_ci.py train-country-embedding-2nd --mode tune --epoch 10 --learning_rate 0.0001
#### check country embedding
python odi/retrain/retrain_ci.py check-country-embedding-2nd










## Retraining (Necessary data loading and Ranking has been done)
### Step 1 - Create one hot encoding for batsman, location and country
python odi/retrain/create_encoding.py batsman --start_date '2014-01-01' --end_date '2018-12-31'

python odi/retrain/create_encoding.py bowler --start_date '2014-01-01' --end_date '2018-12-31'

python odi/retrain/create_encoding.py location --start_date '2014-01-01' --end_date '2018-12-31'

python odi/retrain/create_encoding.py country --start_date '2014-01-01' --end_date '2018-12-31'

(preferably use train period)
####use this to copy locaiton encoding (if new fund in test set which is actually duplicate)
python odi/retrain/create_encoding.py copy --new_value 'Adelaid' --existing_value 'Adelaide Oval'

### Step 2 - Create train-test files for learning country embedding
python odi/retrain/create_train_test.py country-embedding --train_start 2014-01-01 --test_start 2019-01-01

### Step 3 - Learn country embedding
python odi/retrain/retrain.py train-country-embedding
#### tune country embedding
python odi/retrain/retrain.py train-country-embedding --mode tune --epoch 10 --learning_rate 0.0001
#### check country embedding
python odi/retrain/retrain.py check-country-embedding

### Step 4 - Create train-test files for learning batsman embedding
python odi/retrain/create_train_test.py batsman-embedding --train_start 2014-01-01 --test_start 2019-01-01
decide whether to use --include_not_batted
### Step 5 - Learn batsman embedding
python odi/retrain/retrain.py train-batsman-embedding
#### tune batsman embedding
python odi/retrain/retrain.py train-batsman-embedding --mode tune --epoch 10 --learning_rate 0.0001
#### check batsman embedding
python odi/retrain/retrain.py check-batsman-embedding

### Step 6 - Create train test for first innings base model (which is also used for non-embedding feature selection )
python odi/retrain/create_train_test.py first-innings-base --train_start 2015-01-01 --test_start 2019-01-01

### Step 7 - Create first innings base model with non embedding features (as well as non-embedding feature selection)
python odi/retrain/retrain.py select-first-innings-feature-columns

- alternatively can select all columns using

python odi/retrain/retrain.py select-first-innings-feature-columns --select_all True

- optional neural network training
python odi/retrain/retrain.py train-first-innings-base-neural



### Step 8 - Create train test for second innings base model (which is also used for non-embedding feature selection )
python odi/retrain/create_train_test.py second-innings-base --train_start 2015-01-01 --test_start 2019-01-01

### Step 9 - Create second innings base model with non embedding features (as well as non-embedding feature selection)
python odi/retrain/retrain.py select-second-innings-feature-columns

- alternatively can select all columns using

python odi/retrain/retrain.py select-second-innings-feature-columns --select_all True

### Step 10 - Create train test for first innings(including embedding features)
python odi/retrain/create_train_test.py first-innings --train_start 2015-01-01 --test_start 2019-01-01

### Step 11 - Create Train Test for second innings(including embedding features)
python odi/retrain/create_train_test.py second-innings --train_start 2015-01-01 --test_start 2019-01-01

### Step 12 Train first innings(including embedding features)
python odi/retrain/retrain.py first-innings

### Step 13 Train second innings(including embedding features)
python odi/retrain/retrain.py second-innings

### Step 14 Evaluate
python odi/evaluation/evaluate.py first --from_date 2019-01-01 --to_date 2020-12-31 --env dev

python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31 --env dev

#### to use combined without second-innings embedding but base
python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31 --second_innings_emb False --env dev

(this configuration will use final embedding model for first innings and base model for second innings)

python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31 --second_innings_emb False --first_emb_model adversarial --env dev

(this option uses non embedding for second inning and adversarial embedding for first innings on dev)

### Step 15 create Inferencing config 
python odi/inference/inference_config.py --first_innings_emb True --second_innings_emb False

(this configuration will use final embedding model for first innings and base model for second innings)

## Step 16 Create Combined train test
python odi/retrain/create_train_test.py combined-prediction --train_start 2011-01-01 --test_start 2019-01-01

---mix and match embedding
python odi/retrain/create_train_test.py combined-prediction --train_start 2011-01-01 --test_start 2020-12-31 --first_innings_emb False


## Retrain combines
python odi/retrain/retrain.py combined --first_innings_emb True --second_innings_emb True


## Evaluate
python odi/evaluation/evaluate.py first --from_date 2019-01-01 --to_date 2020-12-31

(can use --env dev to evaluate dev models and configurations)

python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31

#### to use in combination mode without second-innings embedding ,but with base models
python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31 --second_innings_emb False

(this configuration will use final embedding model for first innings and base model for second innings)

python odi/evaluation/evaluate.py combined --from_date 2019-01-01 --to_date 2020-12-31 --second_innings_emb False --first_emb_model adversarial --env dev

(this option uses non embedding for second inning and adversarial embedding for first innings on dev)

### to evaluate batsman position
python odi/evaluation/evaluate.py batting-order --from_date 2019-01-01 --to_date 2020-12-31 --env dev

### to evaluate batting recommendation
python odi/evaluation/evaluate.py batting-recommendation --from_date 2019-01-01 --to_date 2019-06-30 --env dev

### to evaluate expected threshold
python odi/evaluation/evaluate.py expected-threshold --from_date 2019-01-01 --to_date 2019-06-30 --env dev

## Retraining batsman run prediction
prerequisite - batsman embedding is already learnt 

### Step 1 create train test for batsman runs
python odi/retrain/create_train_test.py batsman-runs --train_start 2015-01-01 --test_start 2019-01-01

### Step 2 retrain batsman runs model
python odi/retrain/retrain.py batsman-runs

##Train adversarial network
### Step 1 create train test for Adversarial network
python odi/retrain/create_train_test.py adversarial --train_start 2014-01-01 --test_start 2019-01-01

### Step 2 - Learn adversarial embedding
python odi/retrain/retrain.py train-adversarial
#### tune adversarial embedding
python odi/retrain/retrain.py train-adversarial --mode tune --epoch 10 --learning_rate 0.0001
#### check adversarial embedding
python odi/retrain/retrain.py check-adversarial

### Step 3 create train test for Adversarial network
python odi/retrain/create_train_test.py adversarial-first-innings --train_start 2014-01-01 --test_start 2019-01-01

### Step 4 create regression model with adversarial embedding in first innnings
python odi/retrain/retrain.py adversarial-first-innings

# Open issue -  Duplicate stadiums
Melbourne, sydney, Adelaid, Sharjah,Rawalpindi Cricket Stadium
 