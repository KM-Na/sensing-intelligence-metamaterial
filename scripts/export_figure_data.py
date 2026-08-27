"""Export compact data bundles for Figures 2-5.

This script reads the original notebook/result files and writes distribution
data under ``data/``. It does not modify the original repository.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class MinMaxScaler:
    def __init__(self, input_data, output_data):
        self.data_max = jnp.max(input_data, axis=(0, 1), keepdims=True)
        self.data_min = jnp.min(input_data, axis=(0, 1), keepdims=True)
        self.output_data_max = jnp.max(output_data, axis=(0, 1), keepdims=True)
        self.output_data_min = jnp.min(output_data, axis=(0, 1), keepdims=True)

    def transform(self, input_data, output_data):
        data_scale = jnp.where(self.data_max == self.data_min, 1.0, self.data_max - self.data_min)
        output_scale = jnp.where(
            self.output_data_max == self.output_data_min,
            1.0,
            self.output_data_max - self.output_data_min,
        )
        data_normalized = 2.0 * (input_data - self.data_min) / data_scale - 1.0
        output_normalized = (output_data - self.output_data_min) / output_scale
        return data_normalized, output_normalized

    def inverse_transform(self, data_normalized, output_normalized):
        data_original = (data_normalized + 1.0) * (self.data_max - self.data_min) / 2.0 + self.data_min
        output_original = output_normalized * (self.output_data_max - self.output_data_min) + self.output_data_min
        return data_original, output_original


class TimeSeriesCNN(nn.Module):
    features: int = 64
    output_features: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_features, kernel_size=(1,), padding="SAME")(x)
        return x


def as_np(value):
    return np.asarray(value)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def split_dataset(input_data, output_data, seed=0):
    key = jax.random.PRNGKey(seed)
    test_n = input_data.shape[0] // 10
    test_idx = jax.random.randint(key, shape=(test_n,), minval=0, maxval=input_data.shape[0])
    mask = jnp.ones(input_data.shape[0], dtype=bool).at[test_idx].set(False)
    return (
        input_data[mask],
        output_data[mask],
        input_data[test_idx],
        output_data[test_idx],
        test_idx,
    )


def prediction_error(true, pred):
    denom = jnp.sum(true**2)
    return jnp.where(denom == 0, 0.0, jnp.sum((pred - true) ** 2) / denom * 100.0)


def export_fig2(nb_root: Path, out_dir: Path):
    dataset = load_pickle(nb_root / "Simulation/input_source_2_training_dataset_measure_idx_7.pkl")
    record = load_pickle(nb_root / "01152026_cooptimization_sim_1DCNN_data_300_n_source_2_without_contact.pickle")
    retrained = load_pickle(nb_root / "01152026_cooptimization_sim_1DCNN_data_300_n_source_2_retrained_nn.pickle")

    idx = 95
    input_data = dataset["input_data"]
    output_data = dataset["output_data"]
    train_x, train_y, test_x, test_y, test_idx = split_dataset(input_data, output_data)
    scaler = MinMaxScaler(train_x, train_y)
    train_xn, train_yn = scaler.transform(train_x, train_y)
    test_xn, test_yn = scaler.transform(test_x, test_y)

    model = TimeSeriesCNN(output_features=output_data.shape[-1])
    pred_init = model.apply(record["nn_record"][0][0], test_xn)
    _, pred_init_org = scaler.inverse_transform(test_xn, pred_init)

    opt_train_x = record["nn_record"][idx][2]
    opt_train_y = record["nn_record"][idx][3]
    opt_test_x = record["nn_record"][idx][4]
    opt_test_y = record["nn_record"][idx][5]
    opt_train_xn, opt_train_yn = scaler.transform(opt_train_x, opt_train_y)
    opt_test_xn, opt_test_yn = scaler.transform(opt_test_x, opt_test_y)
    pred_opt = model.apply(retrained["params_history"][idx], opt_test_xn)
    _, pred_opt_org = scaler.inverse_transform(opt_test_xn, pred_opt)

    np.savez_compressed(
        out_dir / "figure2_gaussian_impulse.npz",
        time=np.linspace(0.0, 1.0, input_data.shape[1]),
        input_data=as_np(input_data),
        output_data=as_np(output_data),
        test_indices=as_np(test_idx),
        initial_design=as_np(record["design_record"][0]),
        optimized_design=as_np(record["design_record"][idx]),
        design_idx=np.array(idx),
        loss_record=as_np(jnp.array(record["loss_record"])),
        test_input_initial=as_np(test_x),
        test_force_initial=as_np(test_y),
        test_pred_initial=as_np(pred_init_org),
        test_input_optimized=as_np(opt_test_x),
        test_force_optimized=as_np(opt_test_y),
        test_pred_optimized=as_np(pred_opt_org),
        train_input_optimized=as_np(opt_train_x),
        train_force_optimized=as_np(opt_train_y),
        prediction_error_initial=as_np(jax.vmap(prediction_error)(test_y, pred_init_org)),
        prediction_error_optimized=as_np(jax.vmap(prediction_error)(opt_test_y, pred_opt_org)),
    )


def export_fig3(nb_root: Path, out_dir: Path):
    exp_initial = load_pickle(nb_root / "Experiments/Data/111925/left_right_side_training_dataset.pkl")
    exp_opt = load_pickle(nb_root / "Experiments/Data/120225/left_right_side_training_dataset.pkl")
    record = load_pickle(nb_root / "11272025_cooptimization_real_1DCNN_data_300_n_source_2.pickle")

    losses = jnp.array(record["loss_record"])
    min_idx = int(jnp.argmin(losses))
    input_data = exp_initial["experiment_data"]
    output_data = exp_initial["output_data"]
    train_x, train_y, test_x, test_y, test_idx = split_dataset(input_data, output_data)
    scaler = MinMaxScaler(train_x, train_y)
    train_xn, train_yn = scaler.transform(train_x, train_y)
    test_xn, test_yn = scaler.transform(test_x, test_y)

    model = TimeSeriesCNN(output_features=output_data.shape[-1])
    pred_init = model.apply(record["nn_record"][0][0], test_xn)
    _, pred_init_org = scaler.inverse_transform(test_xn, pred_init)

    opt_test_xn = record["nn_record"][min_idx][4]
    opt_test_yn = record["nn_record"][min_idx][5]
    pred_opt = model.apply(record["nn_record"][min_idx][0], opt_test_xn)
    opt_test_x, opt_test_force = scaler.inverse_transform(opt_test_xn, opt_test_yn)
    _, pred_opt_org = scaler.inverse_transform(opt_test_xn, pred_opt)

    np.savez_compressed(
        out_dir / "figure3_experimental_force.npz",
        time=np.linspace(0.0, 1.0, input_data.shape[1]),
        experimental_input_initial=as_np(exp_initial["experiment_data"]),
        experimental_force_initial=as_np(exp_initial["output_data"]),
        simulated_input_initial=as_np(exp_initial["input_data"]),
        experimental_input_optimized=as_np(exp_opt["experiment_data"]),
        experimental_force_optimized=as_np(exp_opt["output_data"]),
        simulated_input_optimized=as_np(exp_opt["input_data"]),
        test_indices=as_np(test_idx),
        initial_design=as_np(record["design_record"][0]),
        optimized_design=as_np(record["design_record"][min_idx]),
        design_idx=np.array(min_idx),
        loss_record=as_np(losses),
        test_input_initial=as_np(test_x),
        test_force_initial=as_np(test_y),
        test_pred_initial=as_np(pred_init_org),
        test_input_optimized=as_np(opt_test_x),
        test_force_optimized=as_np(opt_test_force),
        test_pred_optimized=as_np(pred_opt_org),
        prediction_error_initial=as_np(jax.vmap(prediction_error)(test_y, pred_init_org)),
        prediction_error_optimized=as_np(jax.vmap(prediction_error)(opt_test_force, pred_opt_org)),
    )


def export_fig4(nb_root: Path, out_dir: Path):
    src4 = load_pickle(nb_root / "cooptimization_sim_1DCNN_data_300_n_source_4_all_locations.pickle")
    src8 = load_pickle(nb_root / "cooptimization_sim_1DCNN_data_300_n_source_8_all_locations.pickle")
    np.savez_compressed(
        out_dir / "figure4_sensor_count_losses.npz",
        source4_initial_loss=as_np(src4["init loss"]),
        source4_optimized_loss=as_np(src4["opt loss"]),
        source8_initial_loss=as_np(src8["init loss"]),
        source8_optimized_loss=as_np(src8["opt loss"]),
    )


def export_fig5(nb_root: Path, out_dir: Path):
    results = load_pickle(nb_root / "results_circular_structure_4_sensors_structure_complexity_all.pickle")
    np.savez_compressed(
        out_dir / "figure5_structure_complexity_losses.npz",
        size=as_np(results["size"]),
        initial_prediction_error=as_np(results["init_pred_err_all"]),
        optimized_prediction_error=as_np(results["opt_pred_err_all"]),
        improvement=as_np(results["improvement_all"]),
        number_of_parameters=as_np(results["number of parameters"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "physical-intelligence-change",
        help="Path to the original repository/folder containing the source notebooks and pickles.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    nb_root = args.original_root / "notebooks"
    sys.path.insert(0, str(nb_root))
    sys.path.insert(0, str(args.original_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_fig2(nb_root, args.output_dir)
    export_fig3(nb_root, args.output_dir)
    export_fig4(nb_root, args.output_dir)
    export_fig5(nb_root, args.output_dir)
    print(f"Exported figure data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
