import io
from datetime import datetime
from typing import List

import matplotlib
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import MemoryFile

_FIRST_POSITION = 0
_SECOND_POSITION = 1
_THIRD_POSTION = 2

plt.rcParams['figure.constrained_layout.use'] = True


def plot_full_affected_area_figure(afected_area_image: numpy.array):
    """
    Creates an image of the complete lost and gain area of the selected date windowwith color bars to represent the values
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, height_ratios=[3, 0.2, 0.2], width_ratios=[0.5], layout="constrained")

    custom_cmap_deforestation = (matplotlib.colors.LinearSegmentedColormap.from_list("custom", ['red', 'yellow']))
    custom_cmap_reforestation = (matplotlib.colors.LinearSegmentedColormap.from_list("custom", ['yellow', 'green']))
    divider = make_axes_locatable(ax[_FIRST_POSITION])
    cax_reforestation = divider.append_axes("top", size="5%", pad=0.25)
    cax_deforestation = divider.append_axes("bottom", size="5%", pad=0.05)
    ax[_SECOND_POSITION].axis('off')
    ax[_THIRD_POSTION].axis('off')
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=custom_cmap_deforestation),
                 cax=cax_deforestation, orientation='horizontal', label="Most to Less Deforestade Area", )
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=custom_cmap_reforestation),
                 cax=cax_reforestation, orientation='horizontal', label="Less to Most Recovered Area", )
    cax_deforestation.set_xticks([])
    cax_reforestation.set_xticks([])

    ax[_FIRST_POSITION].imshow(numpy.transpose(afected_area_image, (1, 2, 0)))
    ax[_FIRST_POSITION].axis('off')
    ax[_FIRST_POSITION].set_title('Deforestation and Recovered Green Areas', pad=40)

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(_FIRST_POSITION)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                figure_array = dataset.read()
        memf.close()
        plt.close()

    return figure_array


def plot_single_deforestation_affected_area_figure(afected_area_img_array: numpy.array, difference_date: str):
    """
    Creates a grey scale figure with red pixels that represent the deforestation occured in thoses areas.
    """
    fig, ax = plt.subplots()
    ax.imshow(numpy.transpose(afected_area_img_array, (1, 2, 0)).astype('uint8'))

    legend_elements = [matplotlib.patches.Patch(facecolor='red', label='Deforestaded Area')]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.axis('off')
    plt.title(f'Deforestation Area {difference_date}')

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                deforestation_affected_area_figure_array = dataset.read()
        memf.close()
        plt.close()

    return deforestation_affected_area_figure_array


def plot_deforestation_area_throught_dates_figure(months_list: List[str], list_lost_areas_hac: List[float]):
    """
    Creates a graph that will show the area lost of deforestation in hectares throught the months
    """
    figsize = (len(months_list) * 0.9, len(list_lost_areas_hac) * 0.7)
    plt.figure(figsize=figsize, layout="constrained")
    plt.plot(months_list, list_lost_areas_hac, marker='o', color='red', label='Lost Area in Hectare (Ha)')

    for month, lost_area_hac in zip(months_list, list_lost_areas_hac):
        plt.annotate(f'{lost_area_hac} Ha',
                     (month, lost_area_hac),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontweight='bold'
                     )

    plt.xlabel('Images months')
    plt.ylabel('Affected Area Rate in Hectare (Ha)')
    plt.legend()
    plt.title('Lost Area due to Deforestation Throught Months')
    plt.xticks(months_list, rotation=45)

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                deforestation_area_throught_dates_figure = dataset.read()
        memf.close()
        plt.close()

    return deforestation_area_throught_dates_figure


def plot_lost_gain_area_graph_figure(
        deforestation_area: float, recovered_area: float, initial_date: datetime, final_date: datetime
):
    """
    Creates a bar graph to represent the gain and lost of green area of the user's requested that happended in the date
    window.
    """
    area_in_hectare = (deforestation_area, recovered_area)
    bar_labels = ('Lost Area due to Deforestation', 'Recovered Area')

    plt.bar(
        bar_labels, area_in_hectare, color=('red', 'green'), label=('Lost Area due to Deforestation', 'Recovered Area'),
        align='center'
    )
    for i in range(len(area_in_hectare)):
        plt.annotate('{:.2f} Ha\n'.format(area_in_hectare[i]), xy=(bar_labels[i], area_in_hectare[i]), ha='center',
                     va='center', fontweight='bold')
    plt.xlabel('Affected Area')
    plt.ylabel('Affected Area Rate in Hectare (Ha)')
    plt.legend()
    plt.title(f'Affected Hectares of the Area from {initial_date} -> {final_date}')

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                affected_area_hectare_figure = dataset.read()
        memf.close()
        plt.close()

    return affected_area_hectare_figure


def plot_mean_stdev_max_min_figure(
        months_list: List[str], list_means: List[float], list_stdevs: List[float],
        list_maxs: List[float], list_mins: List[float]
):
    """
    Creates a graph that will show the mean, max, standart deviation and min values throught the months englobed by the
    date window
    """
    figsize = (len(months_list) * 0.5, 6)

    plt.figure(figsize=figsize, layout="constrained")
    plt.plot(months_list, list_means, marker='o', label='Means Values')
    plt.plot(months_list, list_stdevs, marker='o', label='Standart Deviation Values')
    plt.plot(months_list, list_maxs, marker='o', label='Max Values')
    plt.plot(months_list, list_mins, marker='o', label='Min Values')

    plt.xlabel('Images Months')
    plt.ylabel('NDVI Images Values Informations')
    plt.legend()
    plt.title('NDVI Images Informations Throught Months')
    plt.xticks(months_list, rotation=45)
    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                mean_stdev_max_min_figure = dataset.read()
        memf.close()
        plt.close()

    return mean_stdev_max_min_figure
