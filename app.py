import flask
import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = flask.Flask(__name__, template_folder="")

#-------- MODEL GOES HERE -----------#

pipe = pickle.load(open("model/pipe.pkl", "rb"))


#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = request.form
        # print(result)
        date_conv = {"Monday":0,
                    "Tuesday":1,
                    "Wednesday":2,
                    "Thursday":3,
                    "Friday":4,
                    "Saturday":5,
                    "Sunday":6}
        categories = {'Film & Animation': '1',
                     'Autos & Vehicles': '2',
                     'Music': '10',
                     'Pets & Animals': '15',
                     'Sports': '17',
                     'Short Movies': '18',
                     'Travel & Events': '19',
                     'Gaming': '20',
                     'Videoblogging': '21',
                     'People & Blogs': '22',
                     'Comedy': '34',
                     'Entertainment': '24',
                     'News & Politics': '25',
                     'Howto & Style': '26',
                     'Education': '27',
                     'Science & Technology': '28',
                     'Nonprofits & Activism': '29',
                     'Movies': '30',
                     'Anime/Animation': '31',
                     'Action/Adventure': '32',
                     'Classics': '33',
                     'Documentary': '35',
                     'Drama': '36',
                     'Family': '37',
                     'Foreign': '38',
                     'Horror': '39',
                     'Sci-Fi/Fantasy': '40',
                     'Thriller': '41',
                     'Shorts': '42',
                     'Shows': '43',
                     'Trailers': '44'}

    # text = result['title'] + result['tags']
    # text = result["tags"]
    new = pd.DataFrame({'publish_date': int(date_conv[result['date']]),
            'titles': result['title'] + result["vidtags"],
            'category_id': int(categories[result['category']]),
            "title_len": len(result['title'])}, index=[0])

    # print("title: ", result['title'])
    # print("vidtags: ", result['vidtags'])
    # print("title+vidtags ", result['title']+result['vidtags'])
    # print(type(text))

    predicted = pipe.predict(new)[0]
    if predicted == 1:
        prediction = "YOUR LEFT STROKE JUST WENT VIRAL!!"
    else:
        prediction = "Sit down, be humble. Probably not gonna go viral."
    # prediction = '{:,.2f}%'.format(prediction)
    return render_template('index.html', prediction=prediction)




if __name__ == '__main__':
    '''Connects to the server'''

    app.run(port=5000, debug=True)
