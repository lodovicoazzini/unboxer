# How to run the `unboxer`

## Configure the tool

The tool can be configured using the files:

<dl>
    <code>config/config_dirs.py</code>
    <dd>
This file defines the location of the inputs and the outputs of the tool.
</dd>
    <code>config/config_general.py</code>
    <dd>
Ths file defines configurations common to both the heatmaps and the featuremaps.

<code>EXPECTED_LABEL</code>: The label to use to filter the data for the experiment.

<code>MAX_LABELS</code>, <code>MAX_SAMPLES</code>: Respectively the maximum number of rows and columns to use when
exporting the sample images for the clusters.

<code>CLUSTERS_SORT_METRIC</code>: The preference for the clusters when sampling them to show some of their images. If
no sorting is provided, the tool draw a random sample.

<code>CLU_SIM</code>: The similarity metric to use when comparing different clusters.
</dd>
    <code>config/config_heatmaps.py</code>
    <dd>
This fle defines the configuration for the heatmaps.

<code>HEATMAPS_PROCESS_MODE</code>: The processing mode to use when generating the
heatmaps [`LocalLatentMode`, `GlobalLatentMode`].

<code>EXPLAINERS</code>: The list of explainers to use when generating the contributions.

<code>DIM_RED_TECHS</code>: The dimensionality reduction techniques to use to project the contributions in the
two-dimensional latent space. The tool will experiment with the different techniques and choose the best configuration
according to the silhouette score of the corresponding clusters.

<code>CLUS_TECH</code>: The clustering technique to use when grouping the contributions.

<code>ITERATIONS</code>: The number of iterations to use when running the experiment.
</dd>
<code>config/config_featuremaps.py</code>
<dd>
This fle defines the configuration for the featuremaps.

<code>NUM_CELLS</code>: The size of the featuremaps.

<code>BITMAP_THRESHOLD</code>: <strong>COMPLETE</strong>

<code>ORIENTATION_THRESHOLD</code>: <strong>COMPLETE</strong>

<code>FEATUREMAPS_CLUSTERS_MODE</code>: The clustering technique to use on the featuremaps [`ORIGINAL`, `REDUCED`].
</dd>
</dl>

## Generate the heatmaps

You can run the following command to generate the heatmaps.

```commandline
python -m steps.process_featuremaps
```

The tool will experiment with the different explainers, find the best configuration for the dimensionality reduction,
and export the data collected during the experiment.

## Generate the featuremaps

You can run the following command to generate the featuremaps.

```commandline
python -m steps.process_featuremaps
```

The tool will generate the featuremaps, and export the data collected during the experiment.

## Export the insights

You can run the following command to generate the insights about the data.

```commandline
python -m steps.insights.insights
```

**!!! IMPORTANT !!!**<br>
**Remember to generate the heatmaps and the featuremaps before running this command.**

The tool with prompt a menu with a set of options, and will guide you through the process.
