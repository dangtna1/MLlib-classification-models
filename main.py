# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function Definitions
def evaluate_model(model, test_df, label_col="TenYearCHD"):
    """
    Evaluate a given model on the test dataset and calculate metrics.

    Parameters:
    - model: Trained model to evaluate.
    - test_df: Test DataFrame.
    - label_col: Name of the label column.

    Returns:
    - A dictionary containing evaluation metrics.
    """
    # Generate predictions
    predictions = model.transform(test_df)

    # Extract true labels and predictions
    y_test = predictions.select(label_col).rdd.flatMap(lambda x: x).collect()
    y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

    # Print metrics
    for metric, value in metrics.items():
        print(f"Test {metric.capitalize()}: {value}")

    return metrics


# Main Script
if __name__ == "__main__":
    # 1. Start Spark session
    spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

    # 2. Load data
    train_df = spark.read.csv("assessment/train_set.csv", header=True, inferSchema=True)
    test_df = spark.read.csv("assessment/test_set.csv", header=True, inferSchema=True)

    # 3. Assemble features
    feature_cols = [col for col in train_df.columns if col != "TenYearCHD"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Define models and their hyperparameter grids
    models = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(featuresCol="features", labelCol="TenYearCHD"),
            "paramGrid": (ParamGridBuilder()
                          .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0])
                          .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0])
                          .build())
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(featuresCol="features", labelCol="TenYearCHD"),
            "paramGrid": (ParamGridBuilder()
                          .addGrid(DecisionTreeClassifier.maxDepth, [5, 10, 15])
                          .addGrid(DecisionTreeClassifier.maxBins, [32, 64])
                          .build())
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(featuresCol="features", labelCol="TenYearCHD"),
            "paramGrid": (ParamGridBuilder()
                          .addGrid(RandomForestClassifier.numTrees, [10, 50, 100])
                          .addGrid(RandomForestClassifier.maxDepth, [5, 10, 15])
                          .build())
        },
        {
            "name": "Gradient-Boosted Trees",
            "estimator": GBTClassifier(featuresCol="features", labelCol="TenYearCHD"),
            "paramGrid": (ParamGridBuilder()
                          .addGrid(GBTClassifier.maxIter, [10, 50, 100])
                          .addGrid(GBTClassifier.maxDepth, [5, 10, 15])
                          .build())
        }
    ]

    # 4. Define evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="TenYearCHD", metricName="areaUnderROC")

    # Train and evaluate each model
    for model_info in models:
        print(f"Training {model_info['name']}...")
        
        # Build pipeline
        pipeline = Pipeline(stages=[assembler, model_info["estimator"]])
        
        # Set up CrossValidator
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=model_info["paramGrid"],
            evaluator=evaluator,
            numFolds=3,   # 3-fold CV
            parallelism=4 # how many models to train in parallel
        )
        
        # Train model
        cv_model = crossval.fit(train_df)
        
        # Pick best model
        best_model = cv_model.bestModel
        
        # Evaluate on test set
        print(f"Evaluating {model_info['name']}...")
        metrics = evaluate_model(best_model, test_df)
        print(f"Metrics for {model_info['name']}: {metrics}")