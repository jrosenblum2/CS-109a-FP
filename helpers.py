"""
helpers.py - Helper functions for CS109A Final Project Milestone 5

Contents:
- linear_model: Train and evaluate a Linear Regression model.
- forest_model: Train and evaluate Random Forest models with varying numbers of trees.
- knn_model: Train and evaluate a K-Nearest Neighbors (KNN) regression model.
- pcr_model: Train and evaluate a Principal Component Regression (PCR) model.
- poly_model: Train and evaluate Polynomial Regression models with varying polynomial degrees.
- forward_feature_selection_model: Perform Forward Feature Selection using Sequential Feature Selector.
- test_all_models: Test and evaluate multiple models with given training data.
- pairplot: Generate a pair plot to visualize relationships between predictors in a dataset.
- impute_st_rad: Impute stellar radius based on the stellar mass if 'st_rad' is missing.
- impute_st_mass: Impute stellar mass based on the stellar radius if 'st_mass' is missing.
- impute_st_lum: Impute stellar luminosity based on stellar temperature and stellar radius if 'st_lum' is missing.
- scores: Calculate training and validation scores (MSE and R^2) for a given model.
- gradient_boost_model: Train and evaluate Gradient Boosting Regressor models with varying depths.
- xg_boost_model: Train and evaluate XGBoost Regressor models with varying depths.
- random_forest_model: Train and evaluate Random Forest Regressor models with varying parameters.
- bagging_model: Train and evaluate Bagging Regressor models with varying parameters.
- random_forest_and_bagging_model: Compare Random Forest and Bagging Regressor models and visualize their validation MSEs.
"""

# Import libraries for plotting and data manipulation 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import libraries for modeling, testing, etc. 
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Import libraries for AC209 models 
from mlxtend.feature_selection import SequentialFeatureSelector
import xgboost as xgb

# Define constants for conversion during imputation 
sol_rad = 6.957 * 10**8
sol_lum = 3.846 * 10**26
sol_mass = 1.989 * 10**30
earth_flux = 1361

def linear_model(X_train, y_train):
    """
    Train and evaluate a Linear Regression model.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    dict: Dictionary containing evaluation metrics for the Linear Regression model.
    """
    model = LinearRegression().fit(X_train, y_train)

    train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

    MSE = {
        "name": "linear model",
        "size": None,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
    }

    return MSE


def forest_model(X_train, y_train):
    """
    Train and evaluate Random Forest models with varying numbers of trees.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    list: List containing dictionaries with evaluation metrics for each Random Forest model.
    """
    n_trees = [10, 20, 40, 70, 100, 150]
    MSE_list = []

    for i in n_trees:
        model = RandomForestRegressor(
            n_estimators=i, random_state=109, min_samples_leaf=10
        ).fit(X_train, y_train)

        train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

        info = {
            "name": "random_forest",
            "size": i,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
        }

        MSE_list.append(info)

    return MSE_list


def knn_model(X_train, y_train):
    """
    Train and evaluate a K-Nearest Neighbors (KNN) regression model.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    dict: Dictionary containing evaluation metrics for the KNN model.
    """
    # Using cross validated mses to select the best number of neighbors
    ks = range(1, 101)
    # lists to store train and test MSEs
    train_mses = []
    val_mses = []
    # fit a model for each k and record mean cross validated MSEs
    for k in ks:
        cv_scores = cross_validate(
            KNeighborsRegressor(n_neighbors=k),
            X_train,
            y_train,
            cv=5,
            scoring=("neg_mean_squared_error"),
            return_train_score=True,
        )
        train_mses.append(-np.mean(cv_scores["train_score"]))
        val_mses.append(-np.mean(cv_scores["test_score"]))

    best_k_idx = np.argmin(val_mses)
    best_k = ks[best_k_idx]

    model = KNeighborsRegressor(n_neighbors=best_k).fit(X_train, y_train)

    train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

    return {
        "name": "knn",
        "size": f"k={best_k}",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
    }


def pcr_model(X_train, y_train):
    """
    Train and evaluate a Principal Component Regression (PCR) model.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    dict: Dictionary containing evaluation metrics for the PCR model.
    """
    # Perform PCA
    pca = PCA().fit(X_train)
    X_pca_train = pca.transform(X_train)

    # Use mean cross validation MSE to get best number of components to use
    max_components = pca.n_components_
    train_mses = np.empty(max_components)
    val_mses = np.empty_like(train_mses)

    lm = LinearRegression()
    for i, n in enumerate(range(1, max_components + 1)):
        cv = cross_validate(
            lm,
            X_pca_train[:, : n + 1],
            y_train,
            cv=3,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        val_mses[i] = -cv["test_score"].mean()

    # Get best value for n
    best_n = np.argmin(val_mses) + 1

    # Pipeline for the PCR
    pcr_cv = make_pipeline(PCA(best_n), LinearRegression())

    # Fit and get correct values
    model = pcr_cv.fit(X_train, y_train)
    train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

    return {
        "name": "pcr",
        "size": f"n_components = {best_n}",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
    }


def poly_model(X_train, y_train):
    """
    Train and evaluate Polynomial Regression models with varying polynomial degrees.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    list: List containing dictionaries with evaluation metrics for each Polynomial Regression model.
    """
    degrees = range(2, 5)

    info_list = []
    for degree in degrees:
        poly = PolynomialFeatures(degree, include_bias=False)

        X_poly_train = poly.fit_transform(X_train)

        poly_model = LinearRegression().fit(X_poly_train, y_train)

        info = {"name": "polynomial regression", "size": "degree=" + str(degree)}

        train_mse, val_mse, train_r2, val_r2 = scores(poly_model, X_poly_train, y_train)

        info["train_mse"] = train_mse
        info["val_mse"] = val_mse
        info["train_r2"] = train_r2
        info["val_r2"] = val_r2

        info_list.append(info)

    return info_list


def forward_feature_selection_model(X_train, y_train):
    """
    Perform Forward Feature Selection using Sequential Feature Selector.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    dict: Dictionary containing evaluation metrics for the model using Forward Feature Selection.
    """

    # Initialize the features we could want to add to the model
    poly = PolynomialFeatures(3, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    model = LinearRegression()
    max_features = np.shape(X_poly_train)[1]

    sfs = SequentialFeatureSelector(
        model,
        k_features=(1, max_features),
        forward=True,
        floating=False,
        scoring="neg_mean_squared_error",
        cv=5,
    )

    sfs = sfs.fit(X_poly_train, y_train)

    X_train_subset = sfs.transform(X_poly_train)

    model = model.fit(X_train_subset, y_train)

    train_mse, val_mse, train_r2, val_r2 = scores(model, X_train_subset, y_train)

    return {
        "name": "forward feature selection",
        "size": f"n_predictors_selected = {len(sfs.k_feature_idx_)} out of {max_features}",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
    }


def test_all_models(X_train, y_train):
    """
    Test and evaluate multiple models with given training data.

    Parameters:
    X_train: Training features.
    y_train: Target variable for training.

    Returns:
    pandas.DataFrame: DataFrame containing evaluation metrics for various models.
    """
    final_scores = []

    final_scores = final_scores + forest_model(X_train, y_train)
    final_scores = final_scores + poly_model(X_train, y_train)
    final_scores.append(linear_model(X_train, y_train))
    final_scores.append(knn_model(X_train, y_train))
    final_scores.append(pcr_model(X_train, y_train))
    final_scores.append(forward_feature_selection_model(X_train, y_train))

    return pd.DataFrame(final_scores)


def pairplot(X):
    """
    Generate a pair plot to visualize relationships between predictors (features) in a dataset.

    Parameters:
    X (DataFrame): Input DataFrame containing predictors (columns) to be plotted.

    Returns:
    None (displays the pair plot)
    """

    # Select predictors as dataframe columns
    predictors = list(X.columns)
    n = len(predictors)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))

    # Loop through predictors and plot each against each other
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(X[predictors[i]])
                ax.set_xlabel(predictors[i])
                ax.set_ylabel(predictors[i])
            else:
                ax.scatter(X[predictors[j]], X[predictors[i]], alpha=0.8)
                ax.set_xlabel(predictors[j])
                ax.set_ylabel(predictors[i])

            ax.label_outer()

    # Label plots clearly
    plt.suptitle("Plots of Relationships Between Predictors")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def impute_st_rad(row):
    """
    Impute stellar radius based on the stellar mass if 'st_rad' is missing.

    Parameters:
    row (Series): A row from a DataFrame containing 'st_mass' and 'st_rad' columns.

    Returns:
    float: Imputed stellar radius value or original 'st_rad' if not missing.
    """
    if not pd.isna(row["st_mass"]) and pd.isna(row["st_rad"]):
        if row["st_mass"] < 1.66:
            return 1.06 * (row["st_mass"] ** 0.945)
        else:
            return 1.33 * (row["st_mass"] ** 0.555)
    else:
        return row["st_rad"]


def impute_st_mass(row):
    """
    Impute stellar mass based on the stellar radius if 'st_mass' is missing.

    Parameters:
    row (Series): A row from a DataFrame containing 'st_mass' and 'st_rad' columns.

    Returns:
    float: Imputed stellar mass value or original 'st_mass' if not missing.
    """
    if not pd.isna(row["st_rad"]) and pd.isna(row["st_mass"]):
        if row["st_rad"] < 1.71122851495:
            return ((1 / 1.06) * row["st_rad"]) ** (1 / 0.945)
        else:
            return ((1 / 1.33) * row["st_rad"]) ** (1 / 0.555)
    else:
        return row["st_mass"]


def impute_st_lum(row):
    """
    Impute stellar luminosity based on stellar temperature and stellar radius if 'st_lum' is missing.

    Parameters:
    row (Series): A row from a DataFrame containing 'st_rad', 'st_teff', and 'st_lum' columns.

    Returns:
    float: Imputed stellar luminosity value or original 'st_lum' if not missing.
    """

    if (
        not pd.isna(row["st_rad"])
        and not pd.isna(row["st_teff"])
        and pd.isna(row["st_lum"])
    ):
        return np.log10(
            (
                (4 * np.pi)
                * ((row["st_rad"] * sol_rad) ** 2)
                * ((row["st_teff"]) ** 4)
                * (5.67037442 * 10 ** (-8))
            )
            / sol_lum
        )

    else:
        return row["st_lum"]


def scores(model, X_train, y_train):
    """
    Calculate and return training and validation scores (MSE and R^2) for a given model.

    Parameters:
    model: trained sklearn model
    X_train: Training data (predictors).
    y_train: Training data (response).

    Returns:
    tuple: Four values representing training MSE, validation MSE, training R^2, validation R^2.
    """
    mse_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    train_mse = -np.mean(mse_results["train_score"])
    val_mse = -np.mean(mse_results["test_score"])

    r2_results = cross_validate(
        model, X_train, y_train, cv=5, scoring="r2", return_train_score=True
    )
    train_r2 = np.mean(r2_results["train_score"])
    val_r2 = np.mean(r2_results["test_score"])

    return train_mse, val_mse, train_r2, val_r2


def gradient_boost_model(X_train, y_train):
    """
    Train and evaluate Gradient Boosting Regressor models with varying depths.

    Parameters:
    X_train: Training predictors.
    y_train: Response variable for training.

    Returns:
    DataFrame: Information about the best models for each depth tested.
    """
    depths = list(range(1, 5))
    n_estimators = 3000
    best_models_info = []

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(35, 7), sharey=True)
    n_est_graph = list(range(1, n_estimators + 1))

    # Split into k folds for cross-validation 
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Test different depths 
    for depth in depths:
        fold_val_mses = []
        fold_train_mses = []

        # perform manual cross-validation 
        for train_index, val_index in kf.split(X_train):
            X_train_model, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_model, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            gb = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                learning_rate=0.05,
                random_state=0,
            )
            gb.fit(X_train_model, y_train_model)

            predictions_train = list(gb.staged_predict(X_train_model))
            train_mses = [mean_squared_error(y_train_model, predicts) for predicts in predictions_train]

            predictions_val = list(gb.staged_predict(X_val))
            val_mses = [mean_squared_error(y_val, predicts) for predicts in predictions_val]

            fold_train_mses.append(train_mses)
            fold_val_mses.append(val_mses)

        # Average over folds
        avg_train_mses = np.mean(fold_train_mses, axis=0)
        avg_val_mses = np.mean(fold_val_mses, axis=0)

        best_iter_idx = np.argmin(avg_val_mses)

        gb_best = GradientBoostingRegressor(
            n_estimators=best_iter_idx + 1,
            max_depth=depth,
            learning_rate=0.1,
            random_state=0,
        )

        train_mse, val_mse, train_r2, val_r2 = scores(gb_best, X_train, y_train)

        info = {
            "name": "gradient_boosting",
            "max_depth": depth,
            "size": best_iter_idx + 1,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
        }
        best_models_info.append(info)

        axes[depth - 1].plot(n_est_graph, avg_train_mses, label="Training MSE")
        axes[depth - 1].plot(n_est_graph, avg_val_mses, label="Validation MSE")
        axes[depth - 1].axvline(x=best_iter_idx + 1, color="r", label="Best number of estimators")
        axes[depth - 1].set_xlabel("Number of iterations")
        axes[depth - 1].set_ylabel("MSE")
        axes[depth - 1].legend()
        axes[depth - 1].set_title(f"MSE for Gradient Boosting, max_depth = {depth}")
        axes[depth - 1].xaxis.set_tick_params(which="both", labelbottom=True)

    plt.show()
    return pd.DataFrame(best_models_info)


def xg_boost_model(X_train, y_train):
    """
    Train and evaluate XGBoost Regressor models with varying depths.

    Parameters:
    X_train: Training predictors.
    y_train: Response variable for training.

    Returns:
    DataFrame: Information about the best models for each depth tested.
    """
    depths = list(range(1, 7))
    n_estimators = 2000
    best_models_info = []

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(42, 7), sharey=True)
    n_est_graph = list(range(1, n_estimators + 1))

    # Initialize KFold for cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for depth in depths:
        fold_val_mses = []
        fold_train_mses = []

        # Perform cross-validation 
        for train_index, val_index in kf.split(X_train):
            X_train_model, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_model, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=n_estimators,
                                     max_depth=depth,
                                     learning_rate=0.05,
                                     random_state=0)

            model.fit(X_train_model, y_train_model)

            val_mses = []
            train_mses = []

            for i in range(1, n_estimators + 1):
                y_pred_train = model.predict(X_train_model, iteration_range=(1, i))
                train_mses.append(mean_squared_error(y_train_model, y_pred_train))

                y_pred_val = model.predict(X_val, iteration_range=(1, i))
                val_mses.append(mean_squared_error(y_val, y_pred_val))

            fold_train_mses.append(train_mses)
            fold_val_mses.append(val_mses)

        # Average over folds
        avg_train_mses = np.mean(fold_train_mses, axis=0)
        avg_val_mses = np.mean(fold_val_mses, axis=0)

        best_iter_idx = np.argmin(avg_val_mses)

        # Define a new model with the best parameters for this depth
        best_model = xgb.XGBRegressor(objective='reg:squarederror',
                                      n_estimators=best_iter_idx + 1,
                                      max_depth=depth,
                                      learning_rate=0.1,
                                      random_state=0)

        # Get metrics for this best model and append them to the list
        train_mse, val_mse, train_r2, val_r2 = scores(best_model, X_train, y_train)

        info = {
            'name': 'xg_boost',
            'max_depth': depth,
            'size': best_iter_idx + 1,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
        best_models_info.append(info)

        # Make plots of train and validation MSE vs. # iterations, marking the optimal one
        axes[depth - 1].plot(n_est_graph, avg_train_mses, label="Training MSE")
        axes[depth - 1].plot(n_est_graph, avg_val_mses, label="Validation MSE")
        axes[depth - 1].axvline(x=best_iter_idx + 1, color='r', label='Best number of estimators')
        axes[depth - 1].set_xlabel("Number of iterations")
        axes[depth - 1].set_ylabel("MSE")
        axes[depth - 1].legend()
        axes[depth - 1].set_title(f"MSE for XGBoost, max_depth = {depth}");
        axes[depth - 1].xaxis.set_tick_params(which='both', labelbottom=True)

    return pd.DataFrame(best_models_info)


def random_forest_model(X_train, y_train):
    """
    Train and evaluate Random Forest Regressor models with varying parameters.

    Parameters:
    X_train: Training predictors.
    y_train: Response variable for training.

    Returns:
    DataFrame: Information about the models for each hyperparameter combination tested.
    """

    # parameters that we want to test
    n_trees = [10, 20, 40, 70, 100, 150]  # Number of trees
    max_features = list(range(1, 6))  # Size of the random sample of predictors

    MSE_list = []

    for i in n_trees:
        for j in max_features:
            # Train a random forest model
            model = RandomForestRegressor(n_estimators=i, max_features=j)

            # Score the model
            train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

            # Put hyperparamters and their corresponding scores in a dictionary
            info = {
                "name": "random_forest",
                "size": i,  # size refers to the number of trees in the ensemble
                "max_features": j,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
            MSE_list.append(info)

    return pd.DataFrame(MSE_list)


def bagging_model(X_train, y_train):
    """
    Train and evaluate Bagging Regressor models with varying parameters.

    Parameters:
    X_train: Training predictors.
    y_train: Response variable for training.

    Returns:
    DataFrame: Information about the models for each hyperparameter combination tested.
    """

    # parameters that we want to test
    n_trees = [10, 20, 40, 70, 100, 150]  # Number of trees in the model

    MSE_list = []

    for i in n_trees:
        # Create the base decision tree that the bagging model will use
        base_estimator = DecisionTreeRegressor()

        # Create and fit the bagging model on the train data
        model = BaggingRegressor(n_estimators=i)

        train_mse, val_mse, train_r2, val_r2 = scores(model, X_train, y_train)

        info = {
            "name": "bagging",
            "size": i,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
        }
        MSE_list.append(info)

    return pd.DataFrame(MSE_list)


def random_forest_and_bagging_model(X_train, y_train):
    """
    Compare Random Forest and Bagging Regressor models and visualize their validation MSEs.

    Parameters:
    X_train: Training predictors.
    y_train: Response variable for training.

    Returns:
    tuple: DataFrames containing information about Random Forest and Bagging Regressor models.
    """

    # Get information about random forest and bagged models
    random_forest_results = random_forest_model(X_train, y_train)
    bagging_results = bagging_model(X_train, y_train)

    # Plot random forest models at different max_features
    for i in range(1, 6):
        sns.lineplot(
            x="size",
            y="val_mse",
            data=random_forest_results[random_forest_results["max_features"] == i],
            label=f"Max Feautures={i}",
        ).set(
            xlabel="Number of trees",
            ylabel="Validation MSE",
            title="Random Forest Model and Bagging Model Validation MSEs vs Number of Trees",
        )

    # Plot bagging models at different max_features
    sns.lineplot(x="size", y="val_mse", data=bagging_results, label=f"Bagging Model")
    plt.show()

    # Return results for both
    return random_forest_results, bagging_results