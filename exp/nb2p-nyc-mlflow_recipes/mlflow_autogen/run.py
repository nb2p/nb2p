from mlflow.recipes import Recipe

r = Recipe(profile="local")
r.clean()
r.inspect()

r.run("ingest")
r.run("transform")
r.run("train")
r.run("evaluate")
r.run("register")

r.inspect("train")

training_data = r.get_artifact("training_data")
training_data.describe()

trained_model = r.get_artifact("model")
print(trained_model)
