from flask import Flask, render_template, request
from scipy.sparse import data
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(max_df=1,min_df=0.2)
loaded_model = pickle.load(open('model1.pkl', 'rb'))

dataframe = pd.read_csv('fake_news.csv')
dataframe["Label"] = dataframe["Label"].replace(to_replace=0,value="Fake")
dataframe["Label"] = dataframe["Label"].replace(to_replace=1,value="Real")
dataframe = dataframe.drop_duplicates()
dataframe = dataframe.dropna()
x = dataframe['Body']
y = dataframe['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
# pc.fit(tfid_x_train,y_train)

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form["message"]
        pred = fake_news_det(message)
        # print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)