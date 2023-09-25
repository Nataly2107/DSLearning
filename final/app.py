import pandas as pd
from flask import Flask, render_template
import pickle
from prophet.plot import plot_plotly
import plotly.offline as py

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=['GET'])
def home():
    future = model.make_future_dataframe(periods=30, freq='D')
    prediction = model.predict(future)
    fig = plot_plotly(model, prediction)
    py.plot(fig,auto_open=False,filename="./static/temp-plot.html", image_width=1200)

    data = prediction.tail(n=30)
    pred = data[["ds", "trend"]]
    pred["ds"]=pd.to_datetime(pred["ds"]).dt.date
    pred['trend']=pred['trend'].round(2)

    return render_template("index.html", headers=["Дата", "Курс"], data=pred.iloc [1: , :].to_numpy())


if __name__ == "__main__":
    app.run(debug=True)