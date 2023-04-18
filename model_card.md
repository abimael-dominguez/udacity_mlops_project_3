# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    - Person or organization developing model: Abimael Domínguez
    - Model date: 2023-04-17
    - Model version: 0.1.0
    - Model type: Binary classificator (Gaussian Naive Bayes)
    - Information about training algorithms, parameters, fairness constraints or other applied approaches, and features

## Intended Use
    - Prediction task is to determine whether a person makes over 50K a year.
    - Use cases: the model can help to segment people for purpouses such as marketing or public policies to create strategies to address social or economic, issues affecting a community or society.

## Training Data
    - Extraction was done by Barry Becker from the 1994 Census database.
    - Original file: data/census.csv
    - Clean file: data/clean_census.csv


| Column           | Description                                                                                                                   |
|------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Salary           | >50K, <=50K                                                                                                                   |
| age              | continuous                                                                                                                     |
| workclass        | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked                         |
| fnlwgt           | continuous                                                                                                                     |
| education        | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, <br>7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool |
| education-num    | continuous                                                                                                                     |
| marital-status   | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse                      |
| occupation       | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,<br> Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, <br> Protective-serv, Armed-Forces |
| relationship     | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried                                                             |
| race             | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black                                                                     |
| sex              | Female, Male                                                                                                                   |
| capital-gain     | continuous                                                                                                                     |
| capital-loss     | continuous                                                                                                                     |
| hours-per-week   | continuous                                                                                                                     |
| native-country   | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece,<br> South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,<br>  Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, <br>Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. |

- Some special values where found:
    - bad_values = ["?", " ?"]
- Distributions of the clean data:

    ```
    workclass
    ['State-gov' 'Self-emp-not-inc' 'Private' 'Federal-gov' 'Local-gov'
    'Self-emp-inc' 'Without-pay']
    Private             22286
    Self-emp-not-inc     2499
    Local-gov            2067
    State-gov            1279
    Self-emp-inc         1074
    Federal-gov           943
    Without-pay            14
    Name: workclass, dtype: int64

    education
    ['Bachelors' 'HS-grad' '11th' 'Masters' '9th' 'Some-college' 'Assoc-acdm'
    '7th-8th' 'Doctorate' 'Assoc-voc' 'Prof-school' '5th-6th' '10th'
    'Preschool' '12th' '1st-4th']
    HS-grad         9840
    Some-college    6678
    Bachelors       5044
    Masters         1627
    Assoc-voc       1307
    11th            1048
    Assoc-acdm      1008
    10th             820
    7th-8th          557
    Prof-school      542
    9th              455
    12th             377
    Doctorate        375
    5th-6th          288
    1st-4th          151
    Preschool         45
    Name: education, dtype: int64

    marital-status
    ['Never-married' 'Married-civ-spouse' 'Divorced' 'Married-spouse-absent'
    'Separated' 'Married-AF-spouse' 'Widowed']
    Married-civ-spouse       14065
    Never-married             9726
    Divorced                  4214
    Separated                  939
    Widowed                    827
    Married-spouse-absent      370
    Married-AF-spouse           21
    Name: marital-status, dtype: int64

    occupation
    ['Adm-clerical' 'Exec-managerial' 'Handlers-cleaners' 'Prof-specialty'
    'Other-service' 'Sales' 'Transport-moving' 'Farming-fishing'
    'Machine-op-inspct' 'Tech-support' 'Craft-repair' 'Protective-serv'
    'Armed-Forces' 'Priv-house-serv']
    Prof-specialty       4038
    Craft-repair         4030
    Exec-managerial      3992
    Adm-clerical         3721
    Sales                3584
    Other-service        3212
    Machine-op-inspct    1966
    Transport-moving     1572
    Handlers-cleaners    1350
    Farming-fishing       989
    Tech-support          912
    Protective-serv       644
    Priv-house-serv       143
    Armed-Forces            9
    Name: occupation, dtype: int64

    relationship
    ['Not-in-family' 'Husband' 'Wife' 'Own-child' 'Unmarried' 'Other-relative']
    Husband           12463
    Not-in-family      7726
    Own-child          4466
    Unmarried          3212
    Wife               1406
    Other-relative      889
    Name: relationship, dtype: int64

    race
    ['White' 'Black' 'Asian-Pac-Islander' 'Amer-Indian-Eskimo' 'Other']
    White                 25933
    Black                  2817
    Asian-Pac-Islander      895
    Amer-Indian-Eskimo      286
    Other                   231
    Name: race, dtype: int64

    sex
    ['Male' 'Female']
    Male      20380
    Female     9782
    Name: sex, dtype: int64

    native-country
    ['United-States' 'Cuba' 'Jamaica' 'India' 'Mexico' 'Puerto-Rico'
    'Honduras' 'England' 'Canada' 'Germany' 'Iran' 'Philippines' 'Poland'
    'Columbia' 'Cambodia' 'Thailand' 'Ecuador' 'Laos' 'Taiwan' 'Haiti'
    'Portugal' 'Dominican-Republic' 'El-Salvador' 'France' 'Guatemala'
    'Italy' 'China' 'South' 'Japan' 'Yugoslavia' 'Peru'
    'Outlying-US(Guam-USVI-etc)' 'Scotland' 'Trinadad&Tobago' 'Greece'
    'Nicaragua' 'Vietnam' 'Hong' 'Ireland' 'Hungary' 'Holand-Netherlands']
    United-States                 27504
    Mexico                          610
    Philippines                     188
    Germany                         128
    Puerto-Rico                     109
    Canada                          107
    India                           100
    El-Salvador                     100
    Cuba                             92
    England                          86
    Jamaica                          80
    South                            71
    China                            68
    Italy                            68
    Dominican-Republic               67
    Vietnam                          64
    Guatemala                        63
    Japan                            59
    Columbia                         56
    Poland                           56
    Taiwan                           42
    Iran                             42
    Haiti                            42
    Portugal                         34
    Nicaragua                        33
    Peru                             30
    Greece                           29
    Ecuador                          27
    France                           27
    Ireland                          24
    Hong                             19
    Trinadad&Tobago                  18
    Cambodia                         18
    Laos                             17
    Thailand                         17
    Yugoslavia                       16
    Outlying-US(Guam-USVI-etc)       14
    Hungary                          13
    Honduras                         12
    Scotland                         11
    Holand-Netherlands                1
    Name: native-country, dtype: int64

    salary
    ['<=50K' '>50K']
    <=50K    22654
    >50K      7508
    Name: salary, dtype: int64
    ```

## Evaluation Data
    - Datasets:
        - test_size = 20%
        - CV: 5
    - Preprocessing:
        - The program takes in the data, processes it, trains the model, and saves it and the encoder.
        - Most of the preprocessing is performed by the data.py script.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
    - Model performance measures

        - compute_roc_curve(X_test, y_test, best_model)
            - Creates a picture of the ROC curve. This function was tested with a Naive Bayes model from Scikit Learn.
        - compute_model_metrics(y, preds)
            - Validates the trained machine learning model using precision, recall, and F1.
        - compute_metrics_on_slices(data, slice_column, label_column, prediction_column)
            - This function computes performance metrics on subsets of input data for a model's predictions, 
            and returns a list of dictionaries containing the computed metrics for each slice. 
    - Best hyperparameters: {'priors': [0.8, 0.2], 'var_smoothing': 1e-05}

## Overall performance

The overall peformance of the model was:

- Test AUC: 0.6640069292947766
- precision: 0.7545304777594728
- recall: 0.29934640522875816
- fbeta: 0.4286382779597567

## Performance by slices

Note: the metrics are sorted in ascending order and follow the sequence of recall, precision, and fbeta respectively.

### Slices where the model performs worse (top 20)

| feature         | category              | precision | recall | fbeta  |
|----------------|-----------------------|-----------|--------|--------|
| education      | 1st-4th               | 0.0       | 0.0    | 0.0    |
| native-country | Guatemala             | 0.0       | 0.0    | 0.0    |
| native-country | Columbia              | 0.0       | 0.0    | 0.0    |
| marital-status | Married-AF-spouse     | 1.0       | 0.0    | 0.0    |
| native-country | Nicaragua             | 1.0       | 0.0    | 0.0    |
| native-country | Peru                  | 1.0       | 0.0    | 0.0    |
| native-country | Portugal              | 1.0       | 0.0    | 0.0    |
| native-country | Jamaica               | 1.0       | 0.0    | 0.0    |
| native-country | Ireland               | 1.0       | 0.0    | 0.0    |
| native-country | France                | 1.0       | 0.0    | 0.0    |
| native-country | Vietnam               | 1.0       | 0.0    | 0.0    |
| native-country | Poland                | 1.0       | 0.0    | 0.0    |
| native-country | Haiti                 | 1.0       | 0.0    | 0.0    |
| native-country | Hong                  | 1.0       | 0.0    | 0.0    |
| native-country | Cambodia              | 1.0       | 0.0    | 0.0    |
| native-country | Ecuador               | 1.0       | 0.0    | 0.0    |
| native-country | Philippines           | 1.0       | 0.0909 | 0.1667 |
| marital-status | Married-spouse-absent | 0.5       | 0.1111 | 0.1818 |
| native-country | Mexico                | 0.25      | 0.1667 | 0.2    |

### Slices where the model performs better (top 20)

| feature         | category                     | precision | recall  | fbeta   |
|-----------------|------------------------------|-----------|---------|---------|
| workclass       | Without-pay                  | 1.0       | 1.0     | 1.0     |
| education       | Preschool                    | 1.0       | 1.0     | 1.0     |
| native-country  | El-Salvador                  | 1.0       | 1.0     | 1.0     |
| native-country  | South                        | 1.0       | 1.0     | 1.0     |
| native-country  | Dominican-Republic           | 1.0       | 1.0     | 1.0     |
| native-country  | Greece                       | 1.0       | 1.0     | 1.0     |
| native-country  | Yugoslavia                   | 1.0       | 1.0     | 1.0     |
| native-country  | Honduras                     | 1.0       | 1.0     | 1.0     |
| native-country  | Outlying-US(Guam-USVI-etc)   | 1.0       | 1.0     | 1.0     |
| native-country  | Laos                         | 1.0       | 1.0     | 1.0     |
| native-country  | Hungary                      | 1.0       | 1.0     | 1.0     |
| native-country  | Trinadad&Tobago              | 1.0       | 1.0     | 1.0     |
| native-country  | Scotland                     | 1.0       | 1.0     | 1.0     |
| occupation      | Priv-house-serv              | 0.5       | 1.0     | 0.6667  |
| native-country  | China                        | 0.5       | 1.0     | 0.6667  |
| education       | 5th-6th                      | 0.4       | 1.0     | 0.5714  |
| native-country  | Iran                         | 1.0       | 0.6667  | 0.8     |
| native-country  | Puerto-Rico                  | 0.6667    | 0.6667  | 0.6667  |
| education       | 11th                         | 1.0       | 0.5455  | 0.7059  |


## Ethical Considerations
As far as we know, this data uses US dollars, which may not be suitable for certain use cases.
To obtain a more accurate understanding of people's purchasing power, it might be beneficial to use the local currency.

## Caveats and Recommendations
Observations reveal that there are two distinct groups in terms of performance on the "native-country" feature. One group has low performance, while the other has good performance (in the same "native-country" feature). This implies that geographical location may significantly impact the data. To address this issue, it is suggested to either create a model by country or to group similar countries and train a model accordingly. It is also recommended to test different algorithms to improve model performance.