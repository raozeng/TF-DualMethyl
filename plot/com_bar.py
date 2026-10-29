import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_com_bar import main as run_plot_com_bar

if __name__ == "__main__":
    # plot_com_bar.py now directly generates both individual panels and native vector 2x3 merged PDFs/PNGs
    run_plot_com_bar()

