import matplotlib.pyplot as plt
import numpy as np
import os


def showPlots(lostThrust: int):
    """Draw plots based on saved data.

    Note:
            You should have r_hist_ex2.npy (the reference trajectory)
                                            x_lqr_hist_ex2.npy (lqr_trajectory)
                                            x_ad_hist_ex2.npy (adaptive_controller trajectory)
            in the current directory

    """
    r_hist = np.load("r_hist_ex2" + "_" + str(lostThrust) + ".npy")
    r_hist = np.array(r_hist)

    x_hist_lqr = np.load("x_lqr_hist_ex2" + "_" + str(lostThrust) + ".npy")
    x_hist_lqr = np.array(x_hist_lqr)

    x_hist_ad = np.load("x_ad_hist_ex2" + "_" + str(lostThrust) + ".npy")
    x_hist_ad = np.array(x_hist_ad)

    # plot
    s = 2
    plt.figure(1)
    plt.plot(r_hist[:, s], "k", label="Command")
    plt.plot(x_hist_lqr[:, s], label="LQR")
    plt.plot(x_hist_ad[:, s], label="MRAC")
    plt.xlabel("s (seconds)")
    plt.ylabel("m (height)")
    plt.title("LQR vs MRAC vs Reference @" + str(lostThrust * 100) + "% Loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join("plots", "Plot" + "_" + str(lostThrust) + ".png"))
    #     plt.show()
    return
