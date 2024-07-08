import time, os
import xlwt, xlrd
from xlutils.copy import copy
import cv2

def save_info(prompt, img_path, response, folder):
    isFileExist = False
    t = time.strftime("%Y%m%d")
    folder_path = os.path.join(folder, t)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    xls_path =os.path.join(folder_path, t + '.xls')
    print(xls_path)
    if os.path.exists(xls_path) == True:
        isFileExist = True

    dateTime = time.strftime("%Y%m%d-%H.%M.%S")
    
    if isFileExist:
        data = xlrd.open_workbook(xls_path,formatting_info=True)
        excel = copy(wb=data) # 完成xlrd对象向xlwt对象转换     
        excel_table = excel.get_sheet(0) # 获得要操作的页        
        table = data.sheets()[0]       
        nrows = table.nrows # 获得行数        
        ncols = table.ncols # 获得列数
        excel_table.write(nrows, 0, dateTime)
        excel_table.write(nrows, 1, prompt)
        excel_table.write(nrows, 2, img_path)
        # excel_table.write(nrows, 3, str(history_input))
        excel_table.write(nrows, 3, response)
        # excel_table.write(nrows, 5, str(history))
        excel.save(xls_path)
    else:
        f_xls = xlwt.Workbook()
        sheet1 = f_xls.add_sheet('history')
        sheet1.write(0, 0, 'dateTime')
        sheet1.write(0, 1, 'prompt')
        sheet1.write(0, 2, 'img_path')
        sheet1.write(0, 3, 'history_input')
        sheet1.write(0, 4, "response")
        sheet1.write(0, 5, "history")
        sheet1.write(1, 0, dateTime)
        sheet1.write(1, 1, prompt)
        sheet1.write(1, 2, img_path)
        sheet1.write(1, 3, str(history_input))
        sheet1.write(1, 4, response)
        sheet1.write(1, 5, str(history))
        f_xls.save(xls_path)

if __name__=="__main__":
    #img = Image.open("E:\\docker\\aigc\\cubd_portrait_enhance\\res_af.png")
    save_info('123', '', None, folder='./history/')