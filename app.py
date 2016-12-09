"""
This file is part of the flask+d3 Hello World project.
"""
import json
import flask
from flask import request, g
import numpy as np
from superglue_data import get_data


app = flask.Flask(__name__)

segments = {}
data_timeframe = 0
algorithms = {
            'NMF':[('K', int, 10),('eps', float, 3.0)],
            'LDA': [('K', int, 10)],
            'DBSCAN': [('eps', float, 3.0)],
            'HCA':[('K', int, 10)],
            'KMEANS': [('K', int, 10)],
            'HDP':[]
}
@app.before_first_request
def _run_on_start():
    global segments
    global data_timeframe
    data_timeframe = 1
    segments = get_data(data_timeframe)

def update_data(timeframe):
    global segments
    global data_timeframe
    if timeframe!=data_timeframe:
        segments_data = get_data(timeframe)
        data_timeframe  = timeframe
    return segments

@app.route("/")
def gindex():
    """
    When you request the gaus path, you'll get the gaus.html template.
    """
    mux = request.args.get('mux', '')
    muy = request.args.get('muy', '')
    if len(mux)==0: mux="3."
    if len(muy)==0: muy="3."
    algorithms = {
                'NMF':[('K', int, 10),('eps', float, 3.0)],
                'LDA': [('K', int, 10)],
                'DBSCAN': [('eps', float, 3.0)],
                'HCA':[('K', int, 10)],
                'KMEANS': [('K', int, 10)],
                'HDP':[]
    }
    return flask.render_template("gaus.html",mux=mux,muy=muy,
    segments=segments, algorithms=algorithms)


@app.route("/data")
@app.route("/data/<int:timeframe>")
def data(timeframe=1):
    """
    On request, this returns a list of all segments vectors.
    :param timeframe: (optional)
        The number of days to go back in time.
    :returns data:
        A JSON string of the segments vectors.
    """
    global segments
    segments = update_data(timeframe)
    return "number of segments: %d"%len(segments["all_segments"])

@app.route("/gdata")
@app.route("/gdata/<float:mux>/<float:muy>")
def gdata(ndata=100,mux=.5,muy=0.5):
    """
    On request, this returns a list of ``ndata`` randomly made data points.
    about the mean mux,muy
    :param ndata: (optional)
        The number of data points to return.
    :returns data:
        A JSON string of ``ndata`` data points.
    """

    x = np.random.normal(mux,.5,ndata)
    y = np.random.normal(muy,.5,ndata)
    A = 10. ** np.random.rand(ndata)
    c = np.random.rand(ndata)
    return json.dumps([{"_id": i, "x": x[i], "y": y[i], "area": A[i],
        "color": c[i]}
        for i in range(ndata)])

if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    # os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run(port=port)
