from steps.insights.low_level import heatmaps_distance_matrix, clusters_projections, clusters_images, \
    silhouette_by_perplexity

__EXECUTION_DICT = {
    1: heatmaps_distance_matrix,
    2: clusters_projections,
    3: clusters_images,
    4: silhouette_by_perplexity
}

__MENU = """
exit: terminate the program
0: Execute all
1: Distance matrix for the low-level approaches
2: Clusters projections for the low-level approaches
3: Sample cluster images for the low-level approaches
4: Silhouette

Select one or more of the options separated by a space: """

__INVALID_OPTION = lambda message: f'Invalid option [{choices_str}]'

if __name__ == '__main__':
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
