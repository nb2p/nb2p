# Deployment Example: NYC

This example uses the code provided by [MLflow Recipes](https://mlflow.org/docs/latest/recipes.html).

The result is deployed on MLflow server and can be tracked by its Python APIs or `mlflow recipes` CLI commands.

## Folder Structure

### Input

- `data`: The data used by the notebook.
- `notebooks`: The notebooks to construct deployed pipeline.

### Output

- `nb2p_output`: Output pipeline structure from NB2P.
- `mlflow_autogen`: The (auto-generated) MLflow Recipe.
  - `metadata`: The step cards (in HTML format) with interactive pipeline execution log.
- `mlruns`: Metadata for MLflow server.

## Screenshots

### NB2P Execution Process

![](screenshots/01-nb2p_1.png)
![](screenshots/01-nb2p_2.png)
![](screenshots/01-nb2p_3.png)

### MLflow Recipes Execution Step Card

![](screenshots/02-mlflow_card_1.png)
![](screenshots/02-mlflow_card_2.png)
![](screenshots/02-mlflow_card_3.png)
![](screenshots/02-mlflow_card_4.png)
![](screenshots/02-mlflow_card_5.png)
![](screenshots/02-mlflow_card_6.png)
