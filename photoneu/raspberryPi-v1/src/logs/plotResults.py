import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def test():
  df = pd.read_csv("data.log")
  print(df.columns)
  fig = plt.figure()
  ax1 = fig.add_subplot(111, label = "1")
  ax2 = fig.add_subplot(111, label = "2", frame_on = False)
  ax1.scatter(df["motor_x"],df["motor_y"], color = "DarkBlue", label="motor coordenates")
  ax2.scatter(df["cam_x"], df["cam_y"], color = "DarkGreen", label="cam coordenates")
  ax1.set_xlim(df["motor_x"].min(), df["motor_x"].max())
  ax1.set_ylim(df["motor_y"].min(), df["motor_y"].max() )
  ax2.set_xlim(df["cam_x"].min(), df["cam_x"].max())
  ax2.set_ylim(df["cam_y"].min(), df["cam_y"].max())
  ax2.invert_xaxis()
  fig.tight_layout()
  plt.show()

def main():
    df = pd.read_csv("data_1.log")
    print(df.columns)
    for col in df.columns:
#    df["motor_x_norm"] = (df["motor_x"] - df["motor_x"].min())/(df["motor_x"].abs().max() - df["motor_x"].min())
        df[col+str("_norm")] = (df[col] - df[col].min())/(df[col].abs().max() - df[col].min())
    print(df)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label = "1")
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    ax1.scatter(df["motor_x_norm"],df["motor_y_norm"], color = "DarkBlue", label="motor coordenates")
    ax2.scatter(df["cam_x_norm"], df["cam_y_norm"], color = "DarkGreen", label="cam coordenates")
#    ax1.set_xlim(df["motor_x"].min(), df["motor_x"].max())
#    ax1.set_ylim(df["motor_y"].min(), df["motor_y"].max() )
#    ax2.set_xlim(df["cam_x"].min(), df["cam_x"].max())
#    ax2.set_ylim(df["cam_y"].min(), df["cam_y"].max())
    ax2.invert_xaxis()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()