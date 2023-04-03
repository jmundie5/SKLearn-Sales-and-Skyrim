# SKLearn-Sales-and-Skyrim

This project loads and preprocesses two datasets - one related to the game Skyrim and the other related to video game sales - and trains several machine learning models to predict various outcomes.

For the Skyrim dataset, logistic regression is used to predict the race of non-playable characters (NPCs) based on their attributes, and linear regression is used to predict the level of the NPCs based on their attributes.

For the video game sales dataset, logistic regression is used to predict the game developer based on various attributes of the game, and linear regression is used to predict the global sales of the game.

The code also prints out various performance metrics of the trained models, such as the training and test scores, and displays a confusion matrix for the logistic regression models.

The code starts by importing necessary libraries such as numpy, pandas, matplotlib, and sklearn. It also sets some display options for Pandas data frames. The code then loads data from two CSV files, NEWSKYRIMSET.csv and VIDEOGAMESALES.csv, into Pandas data frames. The data in these files is then processed, with the Skyrim data being cleaned up by one-hot encoding the Gender column and removing the Race column for the linear regression analysis. The video game data is cleaned by one-hot encoding the Platform and Genre columns.

The code then applies logistic regression and linear regression models to the processed data from the Skyrim and video game data sets. The logistic regression model predicts the Race of Skyrim non-player characters (NPCs) based on various NPC attributes. The linear regression model predicts the Level of Skyrim NPCs based on the same NPC attributes. For the video game data set, the logistic regression model predicts the game Developer, while the linear regression model predicts the Global Sales.

The model performance is evaluated on both the training and testing sets of data, with the accuracy scores being displayed, along with the confusion matrix for the Skyrim logistic regression model. Finally, some data frames are printed to the console, showing some of the processed data.

![SKResults](https://user-images.githubusercontent.com/81974129/229643859-386c8524-a4ae-4dbc-a597-2ebfbb25e954.PNG)
