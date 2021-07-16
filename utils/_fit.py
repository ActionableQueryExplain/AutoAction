import numpy as np


def fit(data, fit_attributes, agg_attribute, model):

    """
    Fit a model on the given table.
    Args:
        data: Dataframe
            Input table.
        fit_attributes: List
            A set of attributes in original schema to fit the model.
        agg_attribute: String
            The aggregation attribute to fit the prediction model over.
        model:
            The prediction model to fit data
            By default: model = RandomForestClassifier(max_depth=9, n_estimators=500)
    """

    X = np.array(data[fit_attributes]).astype(np.float)
    y = np.array(data[agg_attribute]).flatten().astype(np.float)

    print("Model training start...")
    model = model.fit(X, y)
    print("The model is fit. The prediction score is " + str(model.score(X, y)))

    return X, y, model
