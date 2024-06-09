import numpy as np
import pandas as pd
import time
from scipy  import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

 
def mat_plot(df, reg_df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label = "1",frame_on = False)
    ax2 = fig.add_subplot(111, label = "2",frame_on = False,)
    ax3 = fig.add_subplot(111, label = "3", frame_on = False)
    ax1.scatter(df["motor_x_norm"],df["motor_y_norm"], color = "DarkBlue", label="motor coordenates", alpha=0.7, marker='.')
    ax2.scatter(df["cam_x_norm"], df["cam_y_norm"], color = "DarkRed", label="cam coordenates", alpha=0.7,marker='.')
    ax3.scatter(reg_df["motor_x_norm"], reg_df["motor_y_norm"], color = "DarkGreen", label = "model", alpha = 0.3)
    ax1.set_xlim(df["motor_x"].min(), df["motor_x"].max())
    ax1.set_ylim(df["motor_y"].min(), df["motor_y"].max() )
    ax2.set_xlim(df["cam_x"].min(), df["cam_x"].max())
    ax2.set_ylim(df["cam_y"].min(), df["cam_y"].max())
#   ax2.invert_xaxis()
#   fig.tight_layout()
    fig_2 = plt.figure()
    ax3 = fig_2.add_subplot(211, label = "3")
    ax3.scatter(df["cam_x_norm"], df["motor_x_norm"])
#    ax3.scatter(df["cam_x_norm"], mymodel, color = "DarkGreen")
    ax4 = fig_2.add_subplot(212, label = "4")
    ax4.scatter(df["cam_y_norm"], df["motor_y_norm"])

def main(): 
        name = "data_5.log"
#    for i in range(0,1):
#        name = "data_"
#        name += str(i) + ".log"
        regression(name)
#        df = pd.read_csv(name)
#        df = pd.concat([df, df])

def regression(df_name):
    # read DF
    df = pd.read_csv(df_name)
    
    for col in df.columns:
        df[col+str("_norm")] = (df[col] - df[col].min())/\
            (df[col].abs().max()- df[col].min())
    
    df["cam_x_norm"] = 1 - df["cam_x_norm"]
    X = df[["cam_x_norm", "cam_y_norm"]]

   # linear polynomial regression    
    X_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
   
   # linear regression
    poly_reg_x = LinearRegression()
    poly_reg_x.fit(X_,df["motor_x_norm"]) #X, linear; X_, polynomial
    poly_reg_y = LinearRegression()
    poly_reg_y.fit(X_,df["motor_y_norm"])

    linear_reg_x = LinearRegression()
    linear_reg_x.fit(X,df["motor_x_norm"]) #X, linear; X_, polynomial
    linear_reg_y = LinearRegression()
    linear_reg_y.fit(X,df["motor_y_norm"])
    
   # linear regression 1D
    
    slope_x, intercept_x, r_x, p_x, str_err_x = stats.linregress(df["cam_x_norm"],df["motor_x_norm"])   
    slope_y, intercept_y, r_y, p_y, str_err_y = stats.linregress(df["cam_y_norm"],df["motor_y_norm"])   
    def linear_func_x(x):
        return slope_x * x + intercept_x
    def linear_func_y(x):
        return slope_y * x + intercept_y
    
    df["motor_x_linear"] = linear_reg_x.predict(X)
    df["motor_y_linear"] = linear_reg_y.predict(X)
    df["motor_x_poly"] = poly_reg_x.predict(X_)
    df["motor_y_poly"] = poly_reg_y.predict(X_)
    df["motor_x_stats"] = list(map(linear_func_x, df["cam_x_norm"]))
    df["motor_y_stats"] = list(map(linear_func_y, df["cam_y_norm"]))
    
    df["motor_x_linear_err"] =df["motor_x_linear"] - df["motor_x_norm"]
    df["motor_y_linear_err"] =df["motor_y_linear"] - df["motor_y_norm"]
    df["motor_x_poly_err"] =df["motor_x_poly"] - df["motor_x_norm"]
    df["motor_y_poly_err"] =df["motor_y_poly"] - df["motor_y_norm"]
    df["motor_x_stats_err"] =df["motor_x_stats"] - df["motor_x_norm"]
    df["motor_y_stats_err"] =df["motor_y_stats"] - df["motor_y_norm"]

    print(df)   

    model = sm.OLS(df["motor_x_linear"], X).fit()
    print(model.params)
    model = sm.OLS(df["motor_y_linear"], X).fit()
    print(model.params)
    model = sm.OLS(df["motor_x_poly"], X_).fit()
    print(model.params)
    model = sm.OLS(df["motor_y_poly"], X_).fit()
    print(model.bse)
    print("LR_x sem = {}".format(df["motor_x_linear_err"].sem()))
    print("LR_y sem = {}".format(df["motor_y_linear_err"].sem()))
    print("PR_x sem = {}".format(df["motor_x_poly_err"].sem()))
    print("PR_y sem = {}".format(df["motor_y_poly_err"].sem()))
    print("stats_x sem = {}".format(df["motor_x_stats_err"].sem()))
    print("stats_y sem = {}".format(df["motor_y_stats_err"].sem()))
#    print("LR_x score = {}".format(linear_reg_x.score(X, df["motor_x_linear"])))
#    print("LR_y score = {}".format(linear_reg_y.score(X, df["motor_y_linear"])))
#    print("PR_x score = {}".format(poly_reg_x.score(X_, df["motor_x_poly"])))
#    print("PR_y score = {}".format(poly_reg_y.score(X_, df["motor_y_poly"])))

    print("slope_x = " + str(slope_x))
    print("slope_y = " + str(slope_y))
    print("intercept_x = " + str(intercept_x))
    print("intercept_y = " + str(intercept_y))
    print("Stats LR_x score = {}".format(r_x))
    print("Stats LR_y score = {}".format(r_y))

    #### plot
    ax1 = df.plot(kind="scatter", x="motor_x_norm", y="motor_y_norm", marker="o", \
                    color = "DarkBlue", label="motor coordenates", \
                    alpha = 0.6, figsize=(8,10))
    df.plot(kind="scatter", x="cam_x_norm", y="cam_y_norm", marker="*", \
                    color = "DarkRed", label="cam coordenates", alpha = 0.6, ax= ax1)

    df.plot(kind="scatter", x="motor_x_linear", y="motor_y_linear", marker="+", \
                    color = "DarkGreen", label="linear regression", alpha = 0.8, ax= ax1)
    df.plot(kind="scatter", x="motor_x_poly", y="motor_y_poly", marker="x",\
                    color = "DarkGreen", label="polynomial regression", alpha = 0.8, ax= ax1)
    df.plot(kind="scatter", x="motor_x_stats", y="motor_y_stats", marker=".",\
                    color = "DarkGreen", label="statistical linear regression", alpha = 0.8, ax= ax1)

    major_ticks = np.arange(0, 1.1, 0.125) # 1/16
    minor_ticks = np.arange(0, 1.1, 0.0625)  # 1/8

    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    # And a corresponding grid
    ax1.grid(which='both')

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    plt.title("Regression Techniques Comparison")
    plt.legend(bbox_to_anchor=(0.9, 1.05))
#    plt.grid(visible=True)
#    plt.legend(loc='best')
#    plt.savefig(df_name + "_sklearn_poly_2.png")
    plt.show()

if __name__ == '__main__':
    main()