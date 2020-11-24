from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

class WebInterface:
  def __init__(self, url):
    self.url = url
    self.driver = webdriver.Chrome(ChromeDriverManager().install())
    self.driver.get(self.url)

  def start(self):
    return self.driver.execute_script("window.start();")
    
  def send(self, json):
    return self.driver.execute_script("window.render('" + json + "');")