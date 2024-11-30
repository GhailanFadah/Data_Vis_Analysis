"""analysis.py
Run statistical analyses and plot Numpy ndarray data
Ghailan Fadah
CS 251 Data Analysis Visualization, Spring 2022
"""
from tkinter.font import Font
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        """

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        """
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({"font.size": 18})

    def set_data(self, data):
        """Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        """

        self.data = data
        pass

    def min(self, headers, rows=[]):
        """Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        """

        min_array = np.min(self.data.select_data(headers, rows), axis=0)

        return min_array

    def max(self, headers, rows=[]):
        """Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        """
        max_array = np.max(self.data.select_data(headers, rows), axis=0)

        return max_array

    def range(self, headers, rows=[]):
        """Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        """

        return np.array([self.min(headers, rows), self.max(headers, rows)])

    def mean(self, headers, rows=[]):
        """Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        """

        if len(rows) == 0:
            size = self.data.get_num_samples()
        else:
            size = len(rows)

        array = self.data.select_data(headers, rows)
        sum_array = np.sum(array, axis=0)
        mean_array = np.divide(sum_array, size)

        return mean_array

    def var(self, headers, rows=[]):
        """Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        """

        mean_arr = self.mean(headers, rows)
        data_arr = self.data.select_data(headers, rows)

        arr = np.subtract(data_arr, mean_arr)
        arr_pow = np.power(arr, 2)

        sum = np.sum(arr_pow, axis=0)
        if len(rows) == 0:
            size = self.data.get_num_samples()
        else:
            size = len(rows)

        result = np.multiply(1 / (size - 1), sum)

        return result

    def std(self, headers, rows=[]):
        """Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        """

        return np.sqrt(self.var(headers, rows))

    def show(self):
        """Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        """
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        """Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        """

        x_data = self.data.select_data([ind_var])
        y_data = self.data.select_data([dep_var])

        plt.title(title)
        plt.plot(x_data, y_data, "o")

        return np.squeeze(x_data), np.squeeze(y_data)

    def pair_plot(
        self, data_vars, fig_sz=(12, 12), title="", sharex=False, sharey=False
    ):
        """Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        """

        fig, axis = plt.subplots(
            len(data_vars), len(data_vars), figsize=fig_sz, sharex=sharex, sharey=sharey
        )
        fig.suptitle(title)

        for i in range(len(data_vars)):
            for x in range(len(data_vars)):

                axis[x, i].scatter(
                    self.data.select_data([data_vars[i]]),
                    self.data.select_data([data_vars[x]]),
                )

                if i == 0:
                    axis[x, i].set_ylabel(data_vars[x])

                if x == len(data_vars) - 1:
                    axis[x, i].set_xlabel(data_vars[i])

        return fig, axis
