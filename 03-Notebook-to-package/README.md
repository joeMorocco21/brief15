## From Notebook to package 🎁

It is time to move away from Jupyter Notebook, and start writing reusable code with python packages and modules.

In this challenge we will not implement new functionalities. We will reorganise the existing code into packages and modules.

### Package structure 🗺
### Setup your package ⚙️

├── TaxiFareModel
│   ├── __init__.py
│   ├── data
│   ├── data.py # functions to retrieve and clean data
│   ├── encoders.py #custom encoders and transformers for the pipeline
│   ├── trainer.py # main class that will build and train the pipeline
│   └── utils.py # main class that will build and train the pipeline
├── notebooks




Now that everything is set, let's inspect the content of the provided files...

### `data.py`

You can store  the provided `get_data` and `clean_data` functions.

### `utils.py`

You can store `haversine_vectorized` and `compute_rmse` functions.

You can store the `haversine_distance` function here if you use it.

### `encoders.py`

Let's store the custom encoders and transformers for distance and time features here.

This code file will store all the custom pipeline preprocessing blocks. Meaning the `DistanceTransformer` and `TimeFeaturesEncoder`.

### `trainer.py`

Implement the main class here.

The `Trainer` class is the main class. It should have:
- an `__init__` method called when the class is instanciated
- a `set_pipeline` method that builds the pipeline
- a `run` method that trains the pipeline
- an `evaluate` method evaluating the model

Make sure that you are confident with the following notions:
- attributes and methods of a class
- the `**kwargs` argument of a function and how to use it, (help [HERE](https://www.programiz.com/python-programming/args-and-kwargs) if unclear)



### Test your packaged project

Once you have everything implemented, test that your packaged project works by running:

```bash
python -m TaxiFareModel.trainer
```

Or

```bash
python -i TaxiFareModel/trainer.py
```




👍 Good job!
