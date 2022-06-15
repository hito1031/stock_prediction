from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time

driver = webdriver.Chrome(r'C:\Users\track\Dropbox\My PC (DESKTOP-5UJBDRJ)\Desktop\Machine Learning\株価予測\chromedriver')
'''
options = webdriver.ChromeOptions()
options.add_argument("--headless")

print("connecting to remote browser...")
driver = webdriver.Chrome(ChromeDriverManager().install())
'''
#ここまで一緒

driver.get("https://portal.morningstarjp.com/StockInfo/sec/list?market=0&kind=0000")
driver.find_element_by_xpath("//*[@id='rankingb2']/div[2]/div/p/a[2]").click()
time.sleep(10)
class_elems =  driver.find_elements_by_class_name("tac")
text=[]
for elem in class_elems:
    text.append(elem.text)
del text[0]
dict={}
for t in text:
    