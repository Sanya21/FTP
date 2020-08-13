import sqlalchemy as db
import pandas as pd
import joblib

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://guest:relational@relational.fit.cvut.cz/ftp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)


class Product(db.Model):
    __tablename__ = 'product'

    session_id = db.Column(db.CHAR, primary_key=True)
    sequence_order = db.Column(db.INT)
    category_a = db.Column(db.CHAR)
    category_b = db.Column(db.CHAR)
    category_c = db.Column(db.CHAR)
    category_d = db.Column(db.CHAR)

    def __init__(self, session_id, sequence_order, category_a, category_b, category_c, category_d):
        self.session_id = session_id
        self.sequence_order = sequence_order
        self.category_a = category_a
        self.category_b = category_b
        self.category_c = category_c
        self.category_d = category_d


class Session(db.Model):
    __tablename__ = 'session'

    session_id = db.Column(db.CHAR, primary_key=True)
    start_time = db.Column(db.DATETIME)
    end_time = db.Column(db.DATETIME)
    gender = db.Column(db.CHAR)

    def __init__(self, session_id, start_time, end_time, gender):
        self.session_id = session_id
        self.start_time = start_time
        self.end_time = end_time
        self.gender = gender


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/result')
def result():
    all_data = db.session.query(db.func.sum(Product.sequence_order)).filter_by(session_id=request.values.get('session_id')).all()
    return render_template("result.html", employees=all_data)


@app.route('/predict_result')
def predict_result():
    query = db.session \
        .query(Product.session_id,
               Product.sequence_order,
               Product.category_a,
               Product.category_b,
               Product.category_c,
               Product.category_d,
               Session.start_time,
               Session.end_time,
               Session.gender).outerjoin(Session, Product.session_id == Session.session_id). \
               filter_by(session_id=request.values.get('session_id'))

    model = joblib.load("data/model.pkl")
    category = joblib.load("data/category.pkl")
    df = pd.DataFrame(query)
    df['diff'] = df['end_time'] - df['start_time']
    df['diff'] = df['diff'].astype('timedelta64[s]')
    df['category_a'] = category.get(df['category_a'][0])
    df['category_b'] = category.get(df['category_b'][0])
    df['category_c'] = category.get(df['category_c'][0])
    df['category_d'] = category.get(df['category_d'][0])
    df = df.drop(['start_time', 'end_time', 'session_id', 'gender'], axis=1)
    df['sequence_order'] = df['sequence_order'].astype('int64')
    gender = model.predict(df[0:])
 
    return render_template("predict_result.html", employees=gender[0])


if __name__ == '__main__':
    app.run()


