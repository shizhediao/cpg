#coding:utf-8
#latest website
from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from data_utils import *
from generate_new import Generator
from plan_new import Planner
import datetime
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    rexcel = open_workbook("./results/log.xls")  # 用wlrd提供的方法读取一个excel文件
    row = rexcel.sheets()[0].nrows  # 用wlrd提供的方法获得现在已有的行数
    excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
    table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

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

        #next save log
        now_time = datetime.datetime.now()
        dateFormat = xlwt.XFStyle()
        dateFormat.num_format_str = 'yyyy/mm/dd/hh/mm/ss'
        table.write(row, 0, now_time, dateFormat)
        table.write(row, 1, a)
        output_keywords = keywords[0]+u' '+keywords[1]+u' '+keywords[2]+u' '+keywords[3]
        table.write(row, 2, output_keywords)
        output_poem = ss[0] + u'，' + ss[1] + u'。' + ss[2] + u'，' + ss[3] + u'。'
        table.write(row, 3, output_poem)

        excel.save("./results/log.xls")  # xlwt对象的保存方法，这时便覆盖掉了原来的excel

        return render_template('index.html', K1=keywords[0], K2=keywords[1], K3=keywords[2], K4=keywords[3],
                               S1=ss[0], S2=ss[1], S3=ss[2], S4=ss[3])
    return render_template('index.html')

if __name__ == '__main__':
    #app.jinja_env.auto_reload = True
    #TEMPLATES_AUTO_RELOAD = True
    #SEND_FILE_MAX_AGE_DEFAULT = 0
    #app.debug=False
    app.run(debug=True,port=8888)
