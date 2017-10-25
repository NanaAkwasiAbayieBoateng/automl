
# =======================================================
# | WARNING                                             |   
# | This is not a runnable script, but just an example  |   
# | of how the AutoML pipeline                          |  
# | should look like after development is finished.     |  
# ======================================================= 

from automl.pipeline import Pipeline, Combine, RandomChoice
from automl.model import ModelZoo

# Set the model search space
model_space = ModelSpace([Model1, Model2(lr=10.0)])

# Configure hyperparameter optimization
hyperparameter_optimizer = HyperparameterOptimizer(model_zoo, algorithm='hyperopt') # use default parameter grids for each model type, raise error if none found
# or
param_grid_1 = { 'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
                 'eta' : hp.quniform('eta', 0.025, 0.5, 0.025) }
param_grid_2 = {'l1_reg': hp.choice([1,2,3])}

hyperparameter_optimizer = HyperparameterOptimizer(model_zoo, algorithm='hyperopt', param_grids=[param_grid_1, param_grid_2]) 

# Create AutoML Pipeline
automl_pipeline = Pipeline([
            ('load data', dataset_loader('csv:///path/to/file.csv', cache=True)),
            ('select best features', KTopFeatureFilter(k=10, each=3)),
            ('generate features', RandomChoice([ArithmeticFeatureGenerator(), PolynomialFeatureGenerator()])_,
            ('optimize params', hyperparameter_optimizer),
            ('search for best models', model_space)
           ])

# Also possible
automl_pipeline.add_step(model_zoo) # add any step via api
automl_pipeline.set_metric(auc_score) # manually set metric with any callable(labels, predictions)

# Launch AutoML (distributed out of scope)
preprocess_func, dataset, best_models = automl.run(pipeline, max_steps=10, return_top_n=3, distributed=False)

# Stack results (out of scope)
stacker = ModelStacker(best_models, XGBClassifier(**params))
