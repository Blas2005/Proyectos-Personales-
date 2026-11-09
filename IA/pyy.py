import pyautogui as pg
import time

time.sleep(2)
for i in range(50):
    pg.write("amiga")
    pg.press ('enter')