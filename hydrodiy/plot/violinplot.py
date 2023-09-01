# Code modified from matplotlib demo
# https://matplotlib.org/stable/gallery/statistics/customized_violin.html
import matplotlib.pyplot as plt


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin(ax, data):
    data = np.sort(np.array(data), axis=0)

    parts = ax.violinplot(
            data, showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor("tab:blue")
        pc.set_edgecolor("0.8")
        pc.set_alpha(0.7)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data.T, quartile1, quartile3)])

    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', edgecolor="0.8", \
                        facecolor='white', s=15, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=6)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


