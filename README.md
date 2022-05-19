# 🥡 How to run the `unboxer` 🥡

## 📲 [Install](README-INSTALLATION.md) 📲

## ⚙️ [Configure](README-CONFIGURATION.md) ⚙️

## 🥵 Generate the heatmaps 🥵

You can run the following command to generate the heatmaps.

```commandline
python -m steps.process_featuremaps
```

The tool will experiment with the different explainers, find the best configuration for the dimensionality reduction,
and export the data collected during the experiment.

## 🗺 Generate the featuremaps 🗺

You can run the following command to generate the featuremaps.

```commandline
python -m steps.process_featuremaps
```

The tool will generate the featuremaps, and export the data collected during the experiment.

## 📊 Export the insights 📊

You can run the following command to generate the insights about the data.

```commandline
python -m steps.insights.insights
```

**!!! IMPORTANT !!!**<br>
**Remember to generate the heatmaps and the featuremaps before running this command.**

The tool with prompt a menu with a set of options, and will guide you through the process.

## 🤔 Export the data for the human evaluation 🤔

You can run the following command to export the data for the human evaluation.

```commandline
python -m steps.human_evaluation.human_evaluation
```

**!!! IMPORTANT !!!**<br>
**Remember to generate the heatmaps and the featuremaps before running this command.**

The tool with prompt a menu with a set of options, and will guide you through the process.