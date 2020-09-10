import pandas as pd
import numpy as np
import os
import glob


cwd = os.path.dirname(__file__)

all_files_Desktop = glob.glob(cwd + "/RawKeystrokeData/Desktop/*.csv")
all_files_Phone = glob.glob(cwd + "/RawKeystrokeData/Phone/*.csv")
all_files_Tablet = glob.glob(cwd + "/RawKeystrokeData/Tablet/*.csv")

li_Desktop = []
li_Phone = []
li_Tablet = []


for filename in all_files_Desktop:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_Desktop.append(df)

for filename in all_files_Phone:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_Phone.append(df)

for filename in all_files_Tablet:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_Tablet.append(df)

PhoneFrame = pd.concat(li_Phone, axis=0, ignore_index=True)
DesktopFrame = pd.concat(li_Desktop, axis=0, ignore_index=True)
TabletFrame = pd.concat(li_Tablet, axis=0, ignore_index=True)

