# Disaster Response Pipeline Project


### Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements


### Installation

To install the code the following libraries are needed:
- pandas
- numpy
- sqlalchemy
- nltk
- sklearn
- pickle

Instructions to run the code:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Motivation
The goal of this project is develop a machine learning algorithm that is able to classify disaster related messages into categories in way that allows NGOs, aid workers and other stakeholders to enact quick and adequate measures. 
To achieve this a randomforest-tree classifier was trained with real text data and then fine tuned. 
To allow user input via an interface and make it more user friendly the model was integrated into a flask app using plotly to create visualisations. 


### File Descriptions
The project is structured into the following folders:

- app:contains the run.py and the html templates of the web app
- data: contains the source text files, the script to create the database and the resulting sql database
- model: contains the python script to build the model and the resulting classifier as a pickle file
- the main directory contains two csv files that include different score metrics which are outputted during model evaluation and viusalised on the result page of the web app under model score


### Results
The algorithm performs in general poorly. Only for a single category decent results can be seen. Most of the categories have a f1 score of 0 indicating no classification results at all. 
The algorithm might be improved by using another classifier such as a kn-neighbour or a SVC classifier and creating additional features by combining tokens together.


### Licensing, Authors, Acknowledgements
You can find the Licensing...

..for plotly here:https://github.com/plotly/dash/blob/master/LICENSE





