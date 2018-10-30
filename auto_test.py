#coding:utf-8

from xlrd import open_workbook
from xlutils.copy import copy
from data_utils import *
from generate_new import Generator
from plan_new import Planner
import datetime
import xlwt

'''
rexcel = open_workbook("./results/result.xls") # 用wlrd提供的方法读取一个excel文件
rows = rexcel.sheets()[0].nrows # 用wlrd提供的方法获得现在已有的行数
excel = copy(rexcel) # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
table = excel.get_sheet(0) # 用xlwt对象的方法获得要操作的sheet

values = ["1", "2", "3"]
row = rows
for value in values:
    table.write(row, 0, value) # xlwt对象的写方法，参数分别是行、列、值
    table.write(row, 1, "haha")
    table.write(row, 2, "lala")
    row += 1

excel.save("./results/result.xls") # xlwt对象的保存方法，这时便覆盖掉了原来的excel
'''

'''
initialize origin excel table
for i in range(1, 71):
    file_name = "./test_input/"+str(i)+".txt"
    f = open(file_name,"r")   #设置文件对象
    content = f.read()     #将txt文件的所有内容读入到字符串str中
    print(content)
    rexcel = open_workbook("./results/result.xls")  # 用wlrd提供的方法读取一个excel文件
    excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
    table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet
    table.write(i, 0, i)
    table.write(i, 1, content)
    excel.save("./results/result.xls")  # xlwt对象的保存方法，这时便覆盖掉了原来的excel

    f.close()   #将文件关闭
'''
rexcel = open_workbook("./results/result.xls")  # 用wlrd提供的方法读取一个excel文件
column = rexcel.sheets()[0].ncols  # 用wlrd提供的方法获得现在已有的列数
# rexcel = open_workbook("./results/result.xls")  # 用wlrd提供的方法读取一个excel文件
# print(column)
excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet
now_time = datetime.datetime.now()
dateFormat = xlwt.XFStyle()
dateFormat.num_format_str = 'yyyy/mm/dd/hh/mm/ss'
table.write(0, column, now_time, dateFormat)


for i in range(1, 71):
    file_name = "./test_input/"+str(i)+".txt"
    f = open(file_name,"r")   #设置文件对象
    content = f.read()     #将txt文件的所有内容读入到字符串str中
    print(content)

    planner = Planner()
    generator = Generator()
    line = content.strip()
    if len(line) > 0:
        keywords = planner.plan(line)
        #keywords = line.strip().split()
        print ("Keywords:\t",)
        for word in keywords:
            print (word,)
        print ('\n')
        print ("Poem Generated:\n")
        generator.ya = 0
        generator.yalist = []
        sentences = generator.generate(keywords)
        print ('\t'+sentences[0]+u'，'+sentences[1]+u'。')
        print ('\t'+sentences[2]+u'，'+sentences[3]+u'。')
        print()
    output_poem = sentences[0] + u'，' + sentences[1] + u'。' + sentences[2] + u'，' + sentences[3] + u'。'
    table.write(i, column, output_poem)
    excel.save("./results/result.xls")  # xlwt对象的保存方法，这时便覆盖掉了原来的excel

    f.close()   #将文件关闭