
import numpy as np
import pandas as pd
import time
from scipy  import stats
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import pickle# for saving/loading models

 
def mat_plot(df, reg_df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label = "1",frame_on = False)
    ax2 = fig.add_subplot(111, label = "2",frame_on = False,)
    ax3 = fig.add_subplot(111, label = "3", frame_on = False)
    ax1.scatter(df["motor_x_norm"],df["motor_y_norm"], color = "DarkBlue", label="motor coordinates", alpha=0.7, marker='.')
    ax2.scatter(df["cam_x_norm"], df["cam_y_norm"], color = "DarkRed", label="cam coordinates", alpha=0.7,marker='.')
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
        name = "2024_07_30_03_51_05.log"
#    for i in range(0,1):
#        name = "data_"
#        name += str(i) + ".log"
        df = readDataFrame(name)
#        regression( df )
#        evalRegression(df)
#        plotAndSave(df, name)
#        printModels(df)

#        df = pd.read_csv(name)
#        df = pd.concat([df, df])

def readDataFrame( df_name ):
    # read DF
    df = pd.read_csv(df_name)
    
    for col in df.columns:
        df[col+str("_norm")] = (df[col] - df[col].min())/\
            (df[col].abs().max()- df[col].min())
        print("min in col " + str(col) + ": " + str(df[col].min))
        print("max in col " + str(col) + ": " + str(df[col].max))
    df["cam_x_norm"] = 1 - df["cam_x_norm"]
    return df

def plotAndSave( df, df_name ):
    ax1 = df.plot(kind="scatter", x="motor_x_norm", y="motor_y_norm", marker="o", \
                    color = "DarkBlue", label="motor coordinates", \
                    alpha = 0.6, figsize=(8,10))
    df.plot(kind="scatter", x="cam_x_norm", y="cam_y_norm", marker="*", \
                    color = "DarkRed", label="cam coordinates", alpha = 0.6, ax= ax1)

    df.plot(kind="scatter", x="motor_x_linear", y="motor_y_linear", marker="+", \
                    color = "DarkGreen", label="Multiple LR", alpha = 0.8, ax= ax1)
    df.plot(kind="scatter", x="motor_x_poly", y="motor_y_poly", marker="x",\
                    color = "DarkGreen", label="Polynomial R", alpha = 0.8, ax= ax1)
    df.plot(kind="scatter", x="motor_x_stats", y="motor_y_stats", marker=".",\
                    color = "DarkGreen", label="Linear Regression", alpha = 0.8, ax= ax1)

    major_ticks_x = np.arange(0, 1.0625, 0.125) # 1/16
    major_ticks_y = np.arange(0, 1.325, 0.125) # 1/16
    minor_ticks = np.arange(0, 1.125, 0.0625)  # 1/8

    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.set_yticks(major_ticks_y)
    ax1.set_yticks(minor_ticks, minor=True)
    ax1.set_xlabel("x_normalized")
    ax1.set_ylabel("y_normalized")
    # And a corresponding grid
    ax1.grid(which='both')

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    plt.title("Regression Techniques for Camera Calibration")
#    plt.legend(bbox_to_anchor=(1.15, 0.2))
#    plt.grid(visible=True)
    plt.legend(prop = { "size": 8 }, loc='best')

    plt.savefig(df_name + ".png")
    plt.show()
    df.to_csv(df_name + "_models.csv")
    
def evalRegression(df):
    linear_reg_x = pickle.load(open("linear_x.sav", 'rb'))
    linear_reg_y = pickle.load(open("linear_x.sav", 'rb'))
    X = df[["cam_x_norm", "cam_y_norm"]]
    df["motor_x_linear_eval"] = linear_reg_x.predict(X)
    df["motor_y_linear_eval"] = linear_reg_y.predict(X)
    df["motor_x_eval_linear_err"] =df["motor_x_linear"] - df["motor_x_linear_eval"]
    df["motor_y_eval_linear_err"] =df["motor_y_linear"] - df["motor_y_linear_eval"]
    print(df["motor_x_linear"])
    print(df["motor_x_linear_eval"])
    
def regression(df):
    ''' Regression from cam coordinates to motor coordinates
    '''
    X = df[["cam_x_norm", "cam_y_norm"]]

    #linear regression model 
    linear_reg_x = LinearRegression()
    linear_reg_x.fit(X,df["motor_x_norm"]) #X, linear; X_, polynomial
    linear_reg_y = LinearRegression()
    linear_reg_y.fit(X,df["motor_y_norm"])

    pickle.dump(linear_reg_x, open("linear_x.sav", 'wb'))
    pickle.dump(linear_reg_y, open("linear_y.sav", 'wb'))

    # linear polynomial regression model
    X_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
    poly_reg_x = LinearRegression()
    poly_reg_x.fit(X_,df["motor_x_norm"]) #X, linear; X_, polynomial
    poly_reg_y = LinearRegression()
    poly_reg_y.fit(X_,df["motor_y_norm"])

    pickle.dump(poly_reg_x, open("poly_x.sav", 'wb'))
    pickle.dump(poly_reg_y, open("poly_y.sav", 'wb'))

   # linear regression 1D statistics (sm lib)
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

#    print("Models from statistic sm library:")
#    print("Linear Regression")
#    model = sm.OLS(df["motor_x_linear"], X).fit()
#    print(model.params)
#    print(model.rsquared)
##    print(model.bse) # standard errors of the parameters estimates
#    model = sm.OLS(df["motor_y_linear"], X).fit()
#    print(model.params)
#    print(model.rsquared)
##    print(model.bse) # standard errors of the parameters estimates
#    print("Polynomial Regression")
#    model = sm.OLS(df["motor_x_poly"], X_).fit()
#    print(model.params)
#    print(model.rsquared)
##    print(model.bse) # standard errors of the parameters estimates
#    model = sm.OLS(df["motor_y_poly"], X_).fit()
#    print(model.params)
#    print(model.rsquared)
##    print(model.bse) # standard errors of the parameters estimates
#    print("slope_x = " + str(slope_x))
#    print("slope_y = " + str(slope_y))
#    print("intercept_x = " + str(intercept_x))
#    print("intercept_y = " + str(intercept_y))
#    print("Stats LR_x score = {}".format(r_x))
#    print("Stats LR_y score = {}".format(r_y))
#
def printModels(df):
    print(df)       
    print("LR_x error sem = {}".format(df["motor_x_linear_err"].sem()))   # unbiased standard error of the mean over requested axis
    print("LR_y error sem = {}".format(df["motor_y_linear_err"].sem()))
    print("PR_x error sem = {}".format(df["motor_x_poly_err"].sem()))
    print("PR_y error sem = {}".format(df["motor_y_poly_err"].sem()))
    print("stats_x error sem = {}".format(df["motor_x_stats_err"].sem()))
    print("stats_y error sem = {}".format(df["motor_y_stats_err"].sem()))
#    print("LR_x score = {}".format(linear_reg_x.score(X, df["motor_x_linear"])))
#    print("LR_y score = {}".format(linear_reg_y.score(X, df["motor_y_linear"])))
#    print("PR_x score = {}".format(poly_reg_x.score(X_, df["motor_x_poly"])))
#    print("PR_y score = {}".format(poly_reg_y.score(X_, df["motor_y_poly"])))
    print("motor_x_eval_linear_err = {}".format(df["motor_x_eval_linear_err"].mean()))
    print("motor_y_eval_linear_err = {}".format(df["motor_y_eval_linear_err"].mean()))

    
if __name__ == '__main__':
    main()