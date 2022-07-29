import warnings

from steps.insights import low_level, high_level, combined

__EXECUTION_DICT = {
    1: low_level.heatmaps_distance_matrix,
    2: low_level.heatmaps_clusters_projections,
    3: low_level.heatmaps_clusters_images,
    4: high_level.featuremaps_distance_matrix,
    5: high_level.featuremaps_clusters_projections,
    6: high_level.featuremaps_clusters_images,
    7: combined.combined_distance_matrix
}

__MENU = """
exit: terminate the program
0: Execute all

1: Distance matrix for the heatmaps clusters
2: Clusters projections for the heatmaps clusters
3: Sample clusters images for the heatmaps clusters

4: Distance matrix for the featuremaps clusters
5: Clusters projections for the featuremaps clusters
6: Sample clusters images for the featuremaps clusters

7: Combined distance matrix for the high-level and the low-level clusters

Select one or more of the options separated by a space: """


def __INVALID_OPTION(option):
    return 'Invalid option'


if __name__ == '__main__':
    # Add the choice for execute all
    __EXECUTION_DICT[0] = __EXECUTION_DICT.values()
    warnings.filterwarnings('ignore')
    choices_str = input(__MENU)
    while choices_str != 'exit':
        try:
            choices = [int(choice) for choice in choices_str.split(' ')]

            # Get the handler for the input
            for choice in choices:
                handler = __EXECUTION_DICT.get(choice)
                if handler is not None:
                    try:
                        handler()
                    except TypeError:
                        # Execute all the handlers (skip last as it's the dict values method itself)
                        [handler_item() for handler_item in list(handler)[:-1]]
                else:
                    print(__INVALID_OPTION(choices_str))
        except ValueError:
            print(__INVALID_OPTION(choices_str))

        choices_str = input(__MENU)
