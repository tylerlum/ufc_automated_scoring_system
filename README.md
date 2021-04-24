# ufc_automated_scoring_system

The purpose of this repository is to develop an automated scoring system for UFC fights. When an MMA fight ends without a finish, judges render a decision based on a scoring system that can be quite subjective and prone to error. There have been dozens of high-profile fights with controversial decisions that have upset the MMA community. The goal of this system is to learn an accurate scoring strategy, and then use this to get unbiased fight decisions.

## Performance on Concrete Examples

To make our ML model's results concrete, we show what the models predict on some clear examples. We put these examples in the test set, so that we can train and validate our models on the remaining data, and then analyze the predictions on these examples independently. Note that for this section, we use our deep learning scoring comparison model, described below. These results may change as these models are improved.

### Dominant Wins

Examples of dominant wins include Max Holloway vs. Calvin Kattar, Robert Whittaker vs. Kelvin Gastelum, and Junior dos Santos vs. Cain Velasquez. 

```
Max Holloway score: 20.294231414794922
Calvin Kattar score: 5.815070629119873
Probability that Calvin Kattar won: 5.14968121478887e-07
Actual winner: Max Holloway

Robert Whittaker score: 8.68769359588623
Kelvin Gastelum score: 2.1715078353881836
Probability that Kelvin Gastelum won: 0.0014771156711503863
Actual winner: Robert Whittaker

Junior Dos Santos score: 2.1854841709136963
Cain Velasquez score: 9.674657821655273
Probability that Cain Velasquez won: 0.9994412064552307
Actual winner: Cain Velasquez
```

### Controversial Decisions

Examples of controversial decisions include Daniel Cormier vs. Alexander Gustafsson, Jon Jones vs. Dominick Reyes, Conor McGregor vs. Nate Diaz, Georges St-Pierre vs. Johny Hendricks, Lyoto Machida vs. Mauricio Rua, BJ Penn vs. Frankie Edgar, and more.

```
Daniel Cormier score: 7.0259504318237305
Alexander Gustafsson score: 4.977728843688965
Probability that Alexander Gustafsson won: 0.11423220485448837
Actual winner: Daniel Cormier

Jon Jones score: 3.0689547061920166
Dominick Reyes score: 3.1970248222351074
Probability that Dominick Reyes won: 0.5319738388061523
Actual winner: Jon Jones

Nate Diaz score: 8.818340301513672
Conor McGregor score: 6.82950496673584
Probability that Conor McGregor won: 0.12038014084100723
Actual winner: Conor McGregor

Georges St-Pierre score: 3.760822057723999
Johny Hendricks score: 4.435075759887695
Probability that Johny Hendricks won: 0.6624550223350525
Actual winner: Georges St-Pierre

Lyoto Machida score: 0.0
Mauricio Rua score: 2.7728028297424316
Probability that Mauricio Rua won: 0.9411882758140564
Actual winner: Lyoto Machida

BJ Penn score: 2.3796894550323486
Frankie Edgar score: 2.0094170570373535
Probability that Frankie Edgar won: 0.40847522020339966
Actual winner: Frankie Edgar

Frankie Edgar score: 3.8824844360351562
Gray Maynard score: 4.930335998535156
Probability that Gray Maynard won: 0.7403621077537537
Actual winner: Neither (draw)

Robbie Lawler score: 3.922581672668457
Carlos Condit score: 5.678443431854248
Probability that Carlos Condit won: 0.852690577507019
Actual winner: Robbie Lawler

Johny Hendricks score: 6.012207984924316
Robbie Lawler score: 5.5115556716918945
Probability that Robbie Lawler won: 0.3773874044418335
Actual winner: Robbie Lawler

Jon Jones score: 3.47890043258667
Alexander Gustafsson score: 3.9255313873291016
Probability that Alexander Gustafsson won: 0.6098378896713257
Actual winner: Jon Jones
```

## Supervised learning results

To solve this problem, we test out over 12 different types of ML models to get a sense of the approaches that work best. Note that there has been minimal hyperparameter tuning so far. Many of these models use the default hyperparameters.  

### Linear SVM 

From this initial work, we find that the linear SVM had some of the best performance across these ML models. We achieve a 86% validation accuracy with the default hyperparameters. The confusion matrices and classification reports are shown below.

![Confusion_matrix](https://user-images.githubusercontent.com/26510814/115918527-785fe980-a42c-11eb-9a20-2f0672a034e7.png)

![Classification_Report](https://user-images.githubusercontent.com/26510814/115918533-7a29ad00-a42c-11eb-8948-3ee57b3a5e22.png)

The benefit of these simpler ML models is that we can more easily visualize how they work. Below, we visualize the feature importances that the SVM found, which shows that it found head strikes, significant strikes, control time, and knockdowns to be the most important features.

![Feature_Importances](https://user-images.githubusercontent.com/26510814/115918513-7564f900-a42c-11eb-9223-3c8eb548b63b.png)


### Deep Learning Scoring Comparison Model

Another approach we could take to the problem is to score each individual fighter's fight state (number of knockdowns, significant strikes, takedowns landed in this specific fight), and then compare the scores between the two fighters. The fighter who wins the decision should always have a higher score. Let fighters be labeled A and B. We want a scoring function f(A, B) that returns the winner of the fight. If score(A) > score(B), then A won the fight. To frame this as an ML model, we use a deep learning model that splits the features into fighter 0 and fighter 1 states, scores each fighter, finds the difference in their scores, and then returns the sigmoid of that difference. This ensures consistency, so if f(A, B) = 0, then f(B, A) = 1. A diagram of this model is shown below.

![Deep_Model_Diagram](https://user-images.githubusercontent.com/26510814/115918762-c543c000-a42c-11eb-9136-66c7cccdf5e3.png)

Below, we show the results of this deep learning approach with minimal hyperparameter tuning. This gets reasonable performance with minimal overfitting, and likely can be improved with hyperparameter tuning.

![training](https://user-images.githubusercontent.com/26510814/115920683-6469b700-a42f-11eb-8c92-0a1d9b804b2c.png)

![Confusion_matrix2](https://user-images.githubusercontent.com/26510814/115920690-66337a80-a42f-11eb-83a7-a02d015c1f24.png)

![Classification_Report2](https://user-images.githubusercontent.com/26510814/115920694-67fd3e00-a42f-11eb-912c-db7a155fc9e6.png)


## Unsupervised Learning Visualizations

To better understand the data, we use PCA to find a basis that captures the dataset in a way that can be visualized. Below we visualize two-component PCA dimensionality reduction, with the colors signifying which fighter won the fight. We see that the data starts to look separable using component 1. The variance explained by these components is 0.22014401 and 0.1460742, meaning a total variance explained of 0.36621821. 

![PCA](https://user-images.githubusercontent.com/26510814/115918540-7d249d80-a42c-11eb-99a1-7163e95fae96.png)

Visualizing the components, it appears that component 0 captures the most common fight stats that vary through fights, but does not clearly separate fighter 0 and 1 performances. However, component 1 appears to be the axis that most clearly separates the performances for fighter 0 and 1, with fighter 0 winning when component 1 is positive and fighter 1 winning when component 1 is negative. This primarily focuses on ground strikes, control time, and total strikes.

![PCA_Component0](https://user-images.githubusercontent.com/26510814/115918548-7e55ca80-a42c-11eb-9fd9-cc451349a58f.png)

![PCA_Component1](https://user-images.githubusercontent.com/26510814/115918555-801f8e00-a42c-11eb-89e9-ebcde0ffeade.png)

## Dataset Details

To create this dataset, we leverage the UFC stats website. An example of this data can be found below (link to this page http://ufcstats.com/fight-details/f67aa0b16e16a9ea).

![Website](https://user-images.githubusercontent.com/26510814/115918749-c117a280-a42c-11eb-95a9-da53d1d0e14a.png)

We use a combination of pandas and regex to parse this data and create this raw dataset shown below, which contains the essential stats for a given fight.

![Raw_Data](https://user-images.githubusercontent.com/26510814/115918746-bfe67580-a42c-11eb-8338-4046adb24d90.png)

![Raw_Data2](https://user-images.githubusercontent.com/26510814/115918754-c2e16600-a42c-11eb-9d39-9a71d52401d0.png)

This data is then broken up into an X and y value for the purpose of ML model training.

![Data](https://user-images.githubusercontent.com/26510814/115918759-c4129300-a42c-11eb-948e-46afd403d314.png)

Next, the dataset is broken up into train/validate/test splits. Lastly, the X dataset is augmented with data which has fighter 0 and 1 reversed (so that there is no bias based on how the data is formatted), and then the data is standardized using the X_train mean and standard deviation.

## Files

* UFC_data_scraping.ipynb - Python notebook for scraping data from the UFC stats website.
* UFC_automated_scoring.ipynb - Python notebook for training ML models on the UFC stats
* data - folder of saved files scraped from the UFC stats website

## Next Steps

* Expand the dataset to use round-by-round information and fighter-specific stats (record, age, reach, etc.)
* Visualize the spread of data (histograms of knockdowns, strikes landed, takedowns, etc.)
* Use the judge's score cards for more specific labels than winner and loser
* Predict method of fight ending (decision, KO, submission, etc.)
* Create smarter features for better performance (significant strike difference, etc.)
* Test the classifiers on high-profile, controversial decisions (ensure that these were not used in training or validation)
* Expand this project to allow for fight prediction (given two fighters, predict who will win based on their records and stats)
* Perform detailed hyperparameter tuning analysis (keep independent test set to evaluate afterwards to avoid optimization bias)
* Data augmentation with SMOTE or gaussian noise to increase dataset size
* Integrate boxing datasets for more fights
* Analyze if the remaining error is from aleatoric uncertainty (label noise) or epistemic uncertainty (model insufficient)

