# Disaster Response Pipeline Project


## Hello, This is the 2nd Project in Udacity's Data Science Program of AnTTQ


### Table of content
1. [Project Motivation](#motivation)
2. [Project Requirements](#req)
3. [Project Structure](#files)
4. [Instruction](#instruction)
5. [Licensing, Authors, and Acknowledgements](#licensing)


### Project Overview<a name="motivation"></a>
In this project, I applied learnt Data Engineer, Software Engineer skills to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.

In the data set, it's containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project include also a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data


### Project Requirements <a name="req"></a>
	- NumPy
    - Flask
    - SQLAlchemy
    - SQLite3
    - Pandas
    - Pickle
    - Sklearn
    - Plotly
    - NLTK


### Project Structure <a name="files"></a>
```bash
.
├── app
│   ├── run.py -- the web app file
│   └── templates -- web pages
│       ├── go.html
│       └── master.html 
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py -- ETL
│   └── YourDatabaseName.db
├── models
│   ├── classifier.pkl -- pretrained model
│   └── train_classifier.py -- ML Pipeline 
├── README.md
└── requirements.txt
```
    

### Instructions <a name="instruction"></a>:

1. Run the following commands in the project's root directory to set up your database and model.
	- To setup the environment: 
    	`conda create -n project2`
        `conda activate project2`
        `pip install -r requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`


### Licensing, Authors, Acknowledgements<a name="licensing"></a>
Data's Licensing belonged to Appen. [here](https://appen.com/#data_for_ai).
Many thank to Udacity for the wonderful project.
