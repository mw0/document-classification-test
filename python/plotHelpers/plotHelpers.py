#!/bin/python3

# from sklearn.metrics import confusion_matrix
import timeit
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import matplotlib.colors
from functools import wraps


# Decorator for timing functions.
def timeUsage(func):
    """
    INPUT:
        func	obj, function you want timed

    This is a decorator that prints out the duration of execution for a
    function. Formats differ for cases < 1 min, < 1 hour and >= 1 hour.
    """

    @wraps(func)
    def _timeUsage(*args, **kwds):

        t0 = timeit.default_timer()

        retval = func(*args, **kwds)

        t1 = timeit.default_timer()
        Δt = t1 - t0
        if Δt > 86400.0:
            print(f"Δt: {Δt//86400}d, {int((Δt % 86400)//3600)}h, "
                  f"{int((Δt % 3600)//60)}m, {Δt % 60.0:4.1f}s.")
        elif Δt > 3600.0:
            print(f"Δt: {int(Δt//3600)}h, {int((Δt % 3600)//60)}m, "
                  f"{Δt % 60.0:4.1f}s.")
        elif Δt > 60.0:
            print(f"Δt: {int(Δt//60)}m, {Δt % 60.0:4.1f}s.")
        else:
            print(f"Δt: {Δt % 60.0:5.2f}s.")
        return retval

    return _timeUsage


@timeUsage
def plotConfusionMatrix(confusionMat, xlabels=None, ylabels=None,
                        type='counts', titleText=None, ax=None,
                        saveAs=None, xlabelFontSz=13, ylabelFontSz=13,
                        xtickFontSz=12, ytickFontSz=12, titleFontSz=15,
                        xtickRotate=0.0, ytickRotate=0.0, dir='./'):
    """
    INPUTS:
        confustionMat	np.array (square, floats), containing confusion matrix
        xlabels		list (type=str), containing labels for predict
        ylabels		list (type=str), containing labels for actual
        type		str, in ['counts', 'recall', 'precision'], indicating
                        whether to plot raw counts, normalized along predicted,
                        or normalized along actual
        titleText	str, title for plot
        ax		optional matplotlib.axis object, default: None
        saveAs	str, in ['pdf', 'png', 'svg']
        dir		str, directory into which the plot should be saved,
                        default: './'

    Creates heatmap representing confusion matrix passed via confusionMat. When
    type == 'recall', normalization across predicted values ensures that
    diagonal elements represent recall for each class, and 'precision'
    normalizes across actual values so that diagonal elements represent class
    precisions. (For recall and precision, max values are 1.0.)
    """

    if xlabels is None:
        xlabels = list(range(len(confusionMat) + 1))
    if ylabels is None:
        ylabels = list(range(len(confusionMat) + 1))

    if type == 'counts':
        name = 'Counts'
        fmtType = 'd'
        confusionMatNorm = confusionMat.copy()
    elif type == 'recall':
        np.set_printoptions(precision=2)
        name = 'Recall'
        fmtType = '0.2f'
        confusionMatNorm = (confusionMat.astype('float') /
                            confusionMat.sum(axis=1)[:, np.newaxis])
    elif type == 'precision':
        np.set_printoptions(precision=2)
        name = 'Precision'
        fmtType = '0.2f'
        confusionMatNorm = (confusionMat.astype('float') /
                            confusionMat.sum(axis=0)[np.newaxis, :])

    if titleText is not None:
        titleWords = titleText.split(" ")
        regexSubs = r"[/'\":,\\\/\(\)\[\]\!\—\–\-\.]"
        fileNameAugmentString = "".join([re.sub(regexSubs, '', w)
                                         .lstrip('(').rstrip(')').rstrip(',')
                                         .lstrip(',').capitalize()
                                         for w in titleWords])
        print("fileNameAugmentString:\n", fileNameAugmentString)
        # fileNameAugmentString = "".join([w.replace("/", '').replace("'", '')
        #                                  .replace(":", '').replace(",", '')
        #                                  .lstrip('(').rstrip(')').rstrip(',')
        #                                  .capitalize()
        #                                  for w in titleText.split(" ")])\
        #                           .rstrip('.')
    else:
        fileNameAugmentString = ""

    if ax is None:
        # fig, axis = plt.subplots(1, 1, figsize=(30, 25))
        fig, axis = plt.subplots(1, 1, figsize=(4.7, 5.0))
    else:
        axis = ax
    axis = sns.heatmap(confusionMatNorm, annot=True, fmt=fmtType,
                       cmap=cc.cm.rainbow,
                       xticklabels=xlabels, yticklabels=ylabels, ax=axis)
    axis.set_ylabel('Actual', fontsize=xlabelFontSz)
    axis.set_xlabel('Predicted', fontsize=ylabelFontSz)
    axis.set_title(", ".join(['Confusion matrix', name, titleText]),
                   fontsize=titleFontSz)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=xtickRotate,
                         fontsize=xtickFontSz)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=ytickRotate,
                         fontsize=ytickFontSz)
    plt.tight_layout(rect=[0.0, 0.10, 1.0, 0.90])
    if saveAs == 'pdf':
        plt.savefig(dir + "".join(['ConfusionMatrix', name,
                                   fileNameAugmentString, '.pdf']))
    elif saveAs == 'png':
        plt.savefig(dir + "".join(['ConfusionMatrix', name,
                                   fileNameAugmentString, '.png']))
    elif saveAs == 'svg':
        plt.savefig(dir + "".join(['ConfusionMatrix', name,
                                   fileNameAugmentString, '.svg']))

    return


@timeUsage
def detailedHistogram(data, xlabel=None, ylabel=None,
                      xtickNames=None, ytickNames=None, xlim=None,
                      ylim=None, titleText=None, figName=None, ax=None,
                      ylog=False, saveAs=None, volubility=1, dir='./'):
    """
    INPUTS:
        data		array type, containing values of variable of interest
        xlabel		str, if not None will use to label x-axis, default: None
        ylabel		str, if not None will use to label y-axis, default: None
        xtickNames	array(type=str), if not None will use to label x-ticks,
                        default: None. Axis will be set to have as many ticks.
        ytickNames	array(type=str), if not None will use to label y-ticks,
                        default: None. Axis will be set to have as many ticks.
        xlim, ylim	array type, len=2, for manually setting axis limits,
                        default: None
        titleText	str, if not None, will add to plot, default: None
        figName		str, if not None, will be used for name of figure,
                        default: None
        ax		matplotlib axis object handle, if None will construct
                        plot of size 16 × 8, default: None
        ylog		bool, if True will use log scale on y-axis,
                        default: False
        saveAs		str, in ['pdf', 'png', 'svg'] or None
        volubility	int, more blather for higher values, none for 0,
                        default: 1
        dir		str, directory into which the plot should be saved,
                        default: './'

    Constructs histogram with a bin for every integer on (0, max(data))
    — slow and very detailed, if set(data) is very large.
    """

    if (len(data) <= 1):
        raise ValueError("data must be of array type, len > 1, but "
                         "you provided: ", data)

    freqCt = int(max(data))
    if volubility > 0:
        print(f"Found {freqCt: d} bins in data array.")

    if ax is not None:
        # if not isinstance(ax, matplotlib.axes._subplots.AxesSubplot):
        if not isinstance(ax, matplotlib.axes._axes.Axes):
            raise ValueError("ax is supposed to be a matplotlib axis object, "
                             "but you supplied: ", ax)
        else:
            axis = ax
    else:
        fig, axis = plt.subplots(1, 1, figsize=(18, 6))
        if volubility > 1:
            print("No ax supplied, so creating one.")

    if figName is not None:
        if not isinstance(figName, str):
            raise ValueError("figName must be type str, but you provided: ",
                             figName, ".")
        else:
            nameBase = figName
    else:
        nameBase = 'DetailedHist'
        if volubility > 1:
            print("No figName supplied, so baseName defaulting to "
                  "'Detail/edHist'.")

    if titleText is not None:
        if volubility > 2:
            print("titleText.split(" "):\n", titleText.split(" "))
        titleWords = titleText.split(" ")
        regexSubs = r"[/'\":,\\\/\(\)\[\]\!\—\–\-\.]"
        fileNameAugmentString = "".join([re.sub(regexSubs, '', w)
                                         .lstrip('(').rstrip(')').rstrip(',')
                                         .lstrip(',').capitalize()
                                         for w in titleWords])
        print("fileNameAugmentString:\n", fileNameAugmentString)
    else:
        fileNameAugmentString = ""
    if volubility > 2:
        print(f"Constructed fileNameAugmentString: {fileNameAugmentString}.")

    if xtickNames is not None:
        raise NotImplementedError("Alas, xtickNames feature not "
                                  "yet implimented.")
    if ytickNames is not None:
        raise NotImplementedError("Alas, ytickNames feature not "
                                  "yet implimented.")

    # Start the plotting
    myBins = np.linspace(-0.5, freqCt + 0.5, freqCt + 2)
    freqs, bins, p = axis.hist(data, bins=myBins)

    if ylog:
        axis.set_yscale('log')

    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError("If xlim is not None, it must be set to a list "
                             "of length 2.\nYou provided: ", xlim)
        else:
            axis.set_xlim(xlim)
    if ylim is not None:
        if len(ylim) != 2:
            raise ValueError("If ylim is not None, it must be set to a list "
                             "of length 2.\nYou provided: ", ylim)
        else:
            axis.set_ylim(ylim)

    if xlabel is not None:
        if not isinstance(xlabel, str):
            raise ValueError("xlabel is supposed to be of type str, but you "
                             "supplied: ", xlabel)
        else:
            axis.set_xlabel(xlabel)

    if ylabel is not None:
        if not isinstance(ylabel, str):
            raise ValueError("ylabel is supposed to be of type str, but you "
                             "supplied: ", ylabel)
        else:
            axis.set_ylabel(ylabel)
    else:
        axis.set_ylabel('freqency')

    if titleText is not None:
        if not isinstance(titleText, str):
            raise ValueError("titleText is supposed to be of type str, but you"
                             " supplied: ", titleText)
        else:
            axis.set_title(titleText)

    plt.tight_layout(rect=[0, 0.03, 1, 0.975])

    if saveAs == 'pdf':
        if volubility > 1:
            print("Saving as ",
                  dir + "".join([nameBase, fileNameAugmentString, '.pdf']), '.')
        plt.savefig(dir + "".join([nameBase, fileNameAugmentString, '.pdf']))
    elif saveAs == 'png':
        if volubility > 1:
            print("Saving as ",
                  dir + "".join([nameBase, fileNameAugmentString, '.png']), '.')
        plt.savefig(dir + "".join([nameBase, fileNameAugmentString, '.png']))
    elif saveAs == 'svg':
        if volubility > 1:
            print("Saving as ",
                  dir + "".join([nameBase, fileNameAugmentString, '.svg']), '.')
        plt.savefig(dir + "".join([nameBase, fileNameAugmentString, '.svg']))

    return


@timeUsage
def plotValueCounts(df, colName, barWidth=0.9, figSz=(16.0, 10.0),
                    xrot=65.0, ctRot=90.0, titleText=None, ax=None, saveAs=None,
                    annotateFontSz=15, tickFontSz=15, titleFontSz=17,
                    xlim=None, ylim=None, dir='./'):
    """
    INPUTS:
        df			Pandas DataFrame
        colName		str, column whose item counts are to be histogrammed.
        barWidth	float, fractional width of histogram bars (1.0 for no
                          gaps), default: 0.90
        figSz		tuple (type=float), size of figure in inches, default:
                          (16.0, 10.0)
        xrot		float, angle by which column tick labels are rotated
                	  (x-axis), default: 65.0
        ctRot		float, angle by which value counts are rotated
                 	  (x-axis), default: 90.0
        titleText	str, title for plot
        annotateFontSz	int/float, for numbers above bars
        tickFontSz	int/float, for tick mark labels
        titleFontSz	int/float, for figure title
        ax			optional matplotlib.axis object, default: None
        saveAs		str, in ['pdf', 'png', 'svg']
        xlim, ylim	list (type numeric), if not None will be used to set
                          limits of plot.
        dir		str, directory into which the plot should be saved,
                          default: './'
    """

    classCts = pd.DataFrame(df[colName].value_counts())

    if ax is None:
        ax = classCts.plot(kind='bar', width=barWidth, figsize=figSz,
                           fontsize=tickFontSz, rot=xrot)
    else:
        classCts.plot(kind='bar', width=barWidth, figsize=figSz,
                      fontsize=tickFontSz, rot=xrot, ax=ax)

    rects = ax.patches

    # Number of points between bar and label. Change to your liking.
    space = 5

    # Vertical alignment for positive values
    va = 'bottom'

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"
        else:
            va = "bottom"

        # Use Y value as label and format number with one decimal place
        label = f"{y_value}"

        # Create annotation
        plt.annotate(
            label,                       # use `label` as label
            (x_value, y_value),          # place label at end of the bar
            rotation=ctRot,              # rotate value counts over bars
            fontsize=annotateFontSz,     # size of text over bars
            xytext=(0, space),           # vertically shift label by `space`
            textcoords="offset points",  # interpret `xytext` as offset, points
            ha="center",                 # horizontally center label
            va=va)                       # vertically align for top or bottom

    if xlim is not None:
        if len(xlim) < 2:
            raise ValueError("If xlim is set it must include min and max "
                             "values. You provided: ", xlim)
        else:
            ax.set_xlim(xlim)

    if ylim is not None:
        if len(ylim) < 2:
            raise ValueError("If ylim is set it must include min and max "
                             "values. You provided: ", ylim)
        else:
            ax.set_ylim(ylim)
    if titleText is not None:
        ax.set_title(titleText, fontsize=titleFontSz)
        titleWords = titleText.split(" ")
        regexSubs = r"[/'\":,\\\/\(\)\[\]\!\—\–\-\.]"
        fileNameAugmentString = "".join([re.sub(regexSubs, '', w)
                                         .lstrip('(').rstrip(')').rstrip(',')
                                         .lstrip(',').capitalize()
                                         for w in titleWords])
        print("fileNameAugmentString:\n", fileNameAugmentString)
    else:
        fileNameAugmentString = ""

    if saveAs == 'pdf':
        plt.savefig(dir + "".join([colName + 'Frequencies',
                                   fileNameAugmentString, '.pdf']))
    elif saveAs == 'png':
        plt.savefig(dir + "".join([colName + 'Frequencies',
                                   fileNameAugmentString, '.png']))
    elif saveAs == 'svg':
        plt.savefig(dir + "".join([colName + 'Frequencies',
                                   fileNameAugmentString, '.svg']))

    return


def dependencePlot(ind, shap_values, features, feature_names=None,
                   display_features=None, interaction_index="auto",
                   color="#1E88E5", axis_color="#333333", cmap=None,
                   dot_size=16, x_jitter=0, alpha=1, title=None,
                   xmin=None, xmax=None, ax=None, show=True, dir='./'):
    """
    Create a SHAP dependence plot, colored by an interaction feature.

    *Modification of dependence_plot() from shap module.*

    Plots the value of the feature on the x-axis and the SHAP value of the
    same feature on the y-axis. This shows how the model depends on the given
    feature, and is like a richer extenstion of the classical parital
    dependence plots. Vertical dispersion of the data points represents
    interaction effects. Grey ticks along the y-axis are data points where the
    feature's value was NaN.

    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a
        string it is either the name of the feature to plot, or it can have
        the form "rank(int)" to specify the feature with that rank (ordered by
        mean absolute SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).

    feature_names : list
        Names of the features (length # features).

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead
        of coded values).

    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature
        can also be passed as a string. If "auto" then
        shap.common.approximate_interactions is used to pick what seems to be
        the strongest interaction (note that to find to true stongest
        interaction you need to compute the SHAP interaction values).

    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability
        when feature is discrete.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be
        useful to the show density of the data points when using a large
        dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of
        the format "percentile(float)" to denote that percentile of the
        feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of
        the format "percentile(float)" to denote that percentile of the
        feature's value used on the x-axis.

    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the
         plot will be placed. In this case we do not create a Figure, otherwise
         we do.
    dir: str, directory into which the plot should be saved, default: './'

    """

    if cmap is None:
        cmap = matplotlib.colors.red_blue

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i)
                         for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values, feature_names)

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:                           # off-diag values are split in half
            proj_shap_values = shap_values[:, ind2, :] * 2

        # TODO: remove recursion; generally the functions should be shorter
        # for more maintainable code
        dependence_plot(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=ind2, display_features=display_features, ax=ax,
            show=False, xmin=xmin, xmax=xmax
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1],
                                                          feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        ("'shap_values' and 'features' values must have the same "
         + "number of rows!")
    assert shap_values.shape[1] == features.shape[1], \
        ("'shap_values' must have the same number of columns as 'features'!")

    # Get both the raw and display feature values

    # We randomize the ordering so plotting overlaps are not related to data
    # ordering
    oinds = np.arange(shap_values.shape[0])
    np.random.shuffle(oinds)
    xv = features[oinds, ind].astype(np.float64)
    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # guess what other feature as the stongest interaction with the plotted
    # feature
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, shap_values,
                                                     features)[0]
    interaction_index = convert_name(interaction_index, shap_values,
                                     feature_names)
    categorical_interaction = False

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        cv = features[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(np.float), 5)
        chigh = np.nanpercentile(cv.astype(np.float), 95)
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
            bounds = np.linspace(clow, chigh, int(chigh - clow + 2))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N - 1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(np.sort(xvals)))
            jitter_amount = x_jitter * smallest_diff
            xv += ((np.random.ranf(size=len(xv))*jitter_amount) -
                   (jitter_amount/2))

    # the actual scatter plot, TODO: adapt the dot_size to the number of data
    # points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[oinds, interaction_index].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0,
                       c=cvals[xv_notnan], cmap=cmap, alpha=alpha, vmin=clow,
                       vmax=chigh, norm=color_norm, rasterized=len(xv) > 500)
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if (interaction_index != ind) and (interaction_index is not None):
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(p, ticks=tick_positions)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent()\
                    .transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames], xnames,
                      rotation='vertical', fontsize=11)
    if show:
        with warnings.catch_warnings():  # ignore known matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()


def sortClassificationReport(classificationReport):
    """
    Re-orders output from metrics.classification_report so
    that it is ordered by support, descending.
    """
    tmp = classificationReport.split("\n")
    sortedReport = "\n".join(tmp[:2]) + "\n"
    catValues = []
    for line in tmp[2:-5]:
        items = re.split(r'(\s+)', line)
        newList = [''.join(items[:-8]), ''.join(items[-8:-6]),
                   ''.join(items[-6:-4]), ''.join(items[-4:-2]),
                   ''.join(items[-2:])]

        catValues.append(newList)

    catValues = sorted(catValues, key=lambda v: int(v[4]), reverse=True)

    for repList in catValues:
        sortedReport += (''.join(repList) + "\n")
    sortedReport += "\n".join(tmp[-5:])
    
    return sortedReport
