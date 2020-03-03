import os
import json, time, datetime

input_imgs = '../../lqb_gis/USNI_images/'
files = os.listdir(input_imgs)

for f in files:
    print("old name is:", os.path.join(input_imgs, f))
    t = time.strptime(f, "%B %d, %Y %I:%M %p")
    formated_name = time.strftime("%Y-%m-%d %H:%M:%S", t)
    print("new name is:", os.path.join(input_imgs, formated_name))
    os.rename(os.path.join(input_imgs, f), os.path.join(input_imgs, formated_name))
