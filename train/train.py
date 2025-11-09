from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import pandas as pd
from pathlib import Path
import skl2onnx
import os
from skl2onnx.common.data_types import StringTensorType

# To allow running this script without worrying about the current working directory
# we create an absolute path to the data file based on the script's location
base_path = Path(__file__).resolve().parent
data_path = base_path / "../data" / "train.jsonl"

df = pd.read_json(data_path, lines=True)

X = df["text"].astype(str)
y = df["rating"].astype(int)


# we split the data into training and validation sets
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# we want to create a pipeline that first vectorizes the text data using TF-IDF
# this way we ensure pre-processing is included in the final model
pipe = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                min_df=3,
                ngram_range=(1, 2),  # words
                analyzer="word",
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None,
                class_weight="balanced",
                multi_class="multinomial",
                solver="lbfgs",
            ),
        ),
    ]
)

# train and evaluate, we print the results so they can be logged by the CI system
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_va)
print("Acc:", accuracy_score(y_va, pred))
print("MAE:", mean_absolute_error(y_va, pred))
print(classification_report(y_va, pred))


# Set zipmap=False so probabilities come out as a array instead of a dict
onnx_options = {pipe.named_steps["clf"]: {"zipmap": False}}

# As our model takes string input, it is an 1d array and thus should just be flattened.
initial_types = [("input", StringTensorType([None]))]

onnx_model = skl2onnx.to_onnx(
    pipe,
    initial_types=initial_types,
    options=onnx_options,
    target_opset=skl2onnx.get_latest_tested_opset_version(),  # we go with the latest version, could be versiion locked later
)

# we check and create the output directory if needed
output_dir = f"{base_path}/../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save the model
out_path = f"{output_dir}/model.onnx"
with open(out_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Saved ONNX model to: {out_path}")
