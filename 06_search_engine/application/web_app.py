from flask import Flask, render_template, flash, redirect, request
from forms import SearchForm

from search_engine import preprocess, search

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

""" to use this application you need to download flask:
    pip install flask
    pip install flask-wtf
    
    then by command line go to the directory with the project and write:
    python web_app.py
    
    you can now use the application in the browser:
    http://localhost:5000/search
    
    using the application:
    - click "Prepare Data" button and wait for loading data (preprocessing: making matrix); it can take 10-30 seconds
    - write text which you want to find in the first empty cell
    - write number of results you want to see in the second cell
    - click "Search" button and wait for results
    - after preparing data once you can search as many times as you want
    - in case of any error reload the website or server
"""

matrix = None
dictionary = None


@app.route("/")
@app.route("/search", methods=['GET', 'POST'])
def search_func():
    global matrix, dictionary
    form = SearchForm()

    if request.method == 'POST':
        if 'prepare data' in request.form:
            matrix, dictionary = preprocess()
            flash(f'Data loaded!', 'success')
            return redirect('/search')
        else:
            text = request.form['text']
            k = int(request.form['k'])
            results = search(dictionary, matrix, text, k, return_res=True)
            return render_template('search.html', title='Search', form=form, results=results)

    return render_template('search.html', title='Search', form=form, results=[])


if __name__ == "__main__":
    app.run()
