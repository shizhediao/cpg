#最新网页
from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from data_utils import *
from generate_new import Generator
from plan_new import Planner

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    planner = Planner()
    generator = Generator()
    if request.method == 'POST':
        a = request.form['keywords']
        line = str(a).strip()
        if len(line) > 0:
            keywords = planner.plan(line)
        else:
            k1 = str(request.form['k1'])
            k2 = str(request.form['k2'])
            k3 = str(request.form['k3'])
            k4 = str(request.form['k4'])
            keywords = [k1,k2,k3,k4]
        generator.ya = 0
        generator.yalist = []
        ss = generator.generate(keywords)
        print(keywords)
        print(ss)
        return render_template('index.html', K1=keywords[0], K2=keywords[1], K3=keywords[2], K4=keywords[3],
                               S1=ss[0], S2=ss[1], S3=ss[2], S4=ss[3])
    return render_template('index.html')

if __name__ == '__main__':
    #app.jinja_env.auto_reload = True
    #TEMPLATES_AUTO_RELOAD = True
    #SEND_FILE_MAX_AGE_DEFAULT = 0
    #app.debug=False
    app.run(debug=True,port=8888)
