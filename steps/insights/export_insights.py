import warnings

from steps.insights import low_level, high_level

__EXECUTION_DICT = {
    1: low_level.heatmaps_distance_matrix,
    2: low_level.heatmaps_clusters_projections,
    3: low_level.heatmaps_clusters_images,
    4: low_level.heatmaps_silhouette_by_perplexity,
    5: high_level.featuremaps_distance_matrix,
    6: high_level.featuremaps_clusters_projections,
    7: high_level.featuremaps_clusters_images
}

__MENU = """
exit: terminate the program
0: Execute all

1: Distance matrix for the heatmaps clusters
2: Clusters projections for the heatmaps clusters
3: Sample cluster images for the heatmaps clusters
4: Silhouette distribution by perplexity for the heatmaps approaches

5: Distance matrix for the featuremaps clusters
6: Clusters projections for the featuremaps clusters
7: Sample cluster images for the featuremaps clusters

Select one or more of the options separated by a space: """

__INVALID_OPTION = lambda message: f'Invalid option [{choices_str}]'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    choices_str = input(__MENU)
    while choices_str != 'exit':
        try:
            choices = [int(choice) for choice in choices_str.split(' ')]

            # Get the handler for the input
            for choice in choices:
                handler = __EXECUTION_DICT.get(choice)
                if handler is not None:
                    handler()
                else:
                    print(__INVALID_OPTION(choices_str))
        except ValueError:
            print(__INVALID_OPTION(choices_str))

        choices_str = input(__MENU)
