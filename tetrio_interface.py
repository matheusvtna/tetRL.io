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

    play_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "play-button")))
    play_button.click()

    name_input = self.driver.find_element_by_class_name("swal2-input")
    name_input.send_keys('Robô')

    confirm_button = self.driver.find_element_by_class_name("swal2-confirm")
    confirm_button.click()
  
  def reset(self):
    wait = WebDriverWait(self.driver, 10)

    reset_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "reset-button")))
    reset_button.click()

    name_input = self.driver.find_element_by_class_name("swal2-input")
    name_input.send_keys('Robô')

    confirm_button = self.driver.find_element_by_class_name("swal2-confirm")
    confirm_button.click()

  def press(self, key):
    body = self.driver.find_element_by_tag_name("body")
    body.send_keys(key)

  def step(self):
    return self.driver.execute_script("window.step();")

  def get_grid(self):
    return self.driver.execute_script("return window.grid;")

  def get_block(self):
    return self.driver.execute_script("return window.currentBlock;")

  def get_next_block(self):
    return self.driver.execute_script("return window.nextBlock;")

  def get_player(self):
    return self.driver.execute_script("return window.player;")

  def get_state(self):
    return self.driver.execute_script("return window.currentState;")

  def set_speed(self, speed):
    return self.driver.execute_script("window.setSpeed(" + str(speed) + ");")