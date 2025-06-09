from kerastuner.tuners import RandomSearch
from neuralnetwork import build_tunable_model
import pandas as pd

class ParameterTuner:
    def __init__(self, input_shape, num_classes, project_name="skin_disease_tuning", max_trials=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuner = None

    def run_tuning(self, X_train, y_train, X_val, y_val):
        def build_model_with_batch(hp):
            hp_batch_size = hp.Choice("batch_size", values=[16, 32, 64])
            model = build_tunable_model(hp)
            model._tuned_batch_size = hp_batch_size  # optional: for access later
            return model

        self.tuner = RandomSearch(
            build_model_with_batch,
            objective='val_accuracy',
            max_trials=self.max_trials,
            executions_per_trial=1,
            directory='tuning_logs',
            project_name=self.project_name
        )

        self.tuner.search(
            X_train, y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            batch_size=None  # Keras Tuner will handle it via hp["batch_size"]
        )

        best_model = self.tuner.get_best_models(num_models=1)[0]
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        print("\n Best hyperparameters found:")
        print(f" - Learning rate: {best_hps.get('lr')}")
        print(f" - Batch size: {best_hps.get('batch_size')}")
        for i in range(best_hps.get('conv_layers')):
            print(f" - Conv layer {i+1} filters: {best_hps.get(f'conv_{i}_filters')}")
        print(f" - Dense units: {best_hps.get('dense_units')}")

        return best_model, best_hps

    def export_results(self, filename="tuning_results.csv"):
        """
        Export all trial hyperparameters and their val_accuracy to a CSV file.
        """
        trials = self.tuner.oracle.trials.values()

        if not trials:
            print("No tuning trials found. Did you run tuner.search()?")
            return

        rows = []
        for t in trials:
            row = {**t.hyperparameters.values, "val_accuracy": t.score}
            rows.append(row)

        import pandas as pd
        pd.DataFrame(rows).to_csv(filename, index=False)
        print(f"Tuning results exported to {filename}")

import pandas as pd
import matplotlib.pyplot as plt

def plot_tuning_results(tuner):
    # Step 1: Extract trial data
    trial_data = []
    for trial in tuner.oracle.trials.values():
        hparams = trial.hyperparameters.values
        val_acc = trial.score
        trial_data.append({**hparams, "val_accuracy": val_acc})

    # Step 2: Convert to DataFrame
    df = pd.DataFrame(trial_data)

    # Save ALL combinations, not just top 5
    df.to_csv("tuning_results_all.csv", index=False)

    print(f"\n Saved all {len(df)} combinations to 'tuning_results_all.csv'")

    # Step 3: Print top combinations
    df_sorted = df.sort_values(by="val_accuracy", ascending=False)
    print("\n🔝 Top 5 hyperparameter combinations:")
    print(df_sorted.head())

    # Step 4: Plot some hyperparameter effects
    if "lr" in df.columns:
        plt.figure()
        plt.scatter(df["lr"], df["val_accuracy"])
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Validation Accuracy")
        plt.title("Learning Rate vs Validation Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if "dense_units" in df.columns:
        plt.figure()
        plt.scatter(df["dense_units"], df["val_accuracy"])
        plt.xlabel("Dense Units")
        plt.ylabel("Validation Accuracy")
        plt.title("Dense Units vs Validation Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if "conv_0_filters" in df.columns:
        plt.figure()
        plt.scatter(df["conv_0_filters"], df["val_accuracy"])
        plt.xlabel("Conv Layer 1 Filters")
        plt.ylabel("Validation Accuracy")
        plt.title("Conv Layer 1 Filters vs Validation Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df

