from flask import Flask, render_template

app = Flask(__name__)

# Define routes for different pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/game')
def t():
    return render_template('game.html')
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/play')
def blo():
    return render_template('play.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
