from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

class TetrioInterface:
  def __init__(self, url):
    self.url = url
    self.driver = webdriver.Chrome(ChromeDriverManager().install())

  def navigate(self):
    self.driver.get(self.url)

  def start(self):
    wait = WebDriverWait(self.driver, 10)

    while(True):
      try:
        play_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "play-button")))
        play_button.click()

        name_input = self.driver.find_element_by_class_name("swal2-input")
        name_input.send_keys('Robô')

        confirm_button = self.driver.find_element_by_class_name("swal2-confirm")
        confirm_button.click()

        break
      except:
        continue
      
  
  def reset(self):
    wait = WebDriverWait(self.driver, 10)

    while(True):
      try:
        reset_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "reset-button")))
        reset_button.click()

        name_input = self.driver.find_element_by_class_name("swal2-input")
        name_input.send_keys('Robô')

        confirm_button = self.driver.find_element_by_class_name("swal2-confirm")
        confirm_button.click()

        break
      except:
        continue

  def press(self, key):
    while(True):
      try:
        body = self.driver.find_element_by_tag_name("body")
        body.send_keys(key)
        break
      except:
        continue

  def step(self):
    while(True):
      try:
        return self.driver.execute_script("window.step();")
      except:
        continue

  def get_grid(self):
    while(True):
      try:
        return self.driver.execute_script("return window.grid;")
      except:
        continue

  def get_block(self):
    while(True):
      try:
        return self.driver.execute_script("return window.currentBlock;")
      except:
        continue

  def get_next_block(self):
    while(True):
      try:
        return self.driver.execute_script("return window.nextBlock;")
      except:
        continue

  def get_player(self):
    while(True):
      try:
        return self.driver.execute_script("return window.player;")
      except:
        continue

  def get_state(self):
    while(True):
      try:
        return self.driver.execute_script("return window.currentState;")
      except:
        continue

  def set_speed(self, speed):
    while(True):
      try:
        return self.driver.execute_script("window.setSpeed(" + str(speed) + ");")
      except:
        continue