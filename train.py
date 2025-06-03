"""Train a model with W and top nodes including classification of edges """
from pathlib import Path

# To Run the Topograph
#
# Start by cmsenv in CMSSW_13_2_10
# Then, set up the environment with the closest variables using the command below
# source /cvmfs/sft.cern.ch/lcg/views/LCG_103_LHCB_7/x86_64-centos7-clang12-opt/setup.sh
# tensorflow 2.8.0 (2.10 not an option)
#
# Then run fully or partially with the command below
# python3 train.py configs/config_full.yaml Outputs --test
# python3 train.py configs/config_partial.yaml Outputs --test
# python3 train.py configs/config_full.yaml Outputs --test_file
# python3 train.py configs/config_partial.yaml Outputs --test_file b_test.h5 --all

import joblib
import numpy as np
import os
import shutil
from functions.functions import get_best_model_weights, load_and_mask_data
from functions.model_topograph import TopographModel
from functions.tools import Configuration, CustomParser


def get_parser():
    """Get arguments set on the command line."""
    parser = CustomParser(description="Options for NN Training")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Only run for 2 epochs and limited statistics "
        + "to test newly implemented features",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Only predict with the trained model.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Predict on all available events not just matchable events.",
    )
    parser.add_argument("--test_file", action="store", help="Test file to predict on.")
    parser.add_argument(
        "--jet_scale",
        action="store_true",
        help="Scale the jet energy up by 2.5%% when predicting on the test set.",
    )
    parser.add_argument(
        "--jet_resolution",
        action="store_true",
        help="Smear the jet energy by 5%% when predicting on the test set.",
    )
    parsed_args = parser.parse_args()

    return parsed_args


def train(config):
    """
    Load the training/validation data, build the model and fit it.
    """
    kwargs = {
        "jet_indices": True,
        "jets": True,
        "min_n_jets": 6,
        "max_n_jets": 16,
        "n_events": config["n_events"],
        "partons_top": True,
    }
    train_dataset = load_and_mask_data(
        config["train_file"], matchability=config["matchability"], **kwargs
    )
    train_dataset.calc_truth_edges_top()
    train_dataset.calc_parton_mask()
    if config["matchability"] is False:
        train_dataset.mask_fully_impossible_events()
    train_dataset.preprocess_fit(
        config["log_dir"], same_scaler_everything=config["same_scaler_everything"]
    )

    val_dataset = load_and_mask_data(
        config["val_file"], matchability=config["matchability"], **kwargs
    )
    val_dataset.calc_truth_edges_top()
    val_dataset.calc_parton_mask()
    if config["matchability"] is False:
        val_dataset.mask_fully_impossible_events()
    val_dataset.preprocess_fit(
        config["log_dir"],
        same_scaler_everything=config["same_scaler_everything"],
        jet_scaler=train_dataset.jets.scaler,
        top_scaler=train_dataset.top_partons.scaler,
    )
    
    model = TopographModel(config)
    model.build(
        [
            (1,) + train_dataset.jets.input_shape(config["use_flavour_tagging"])[1:],
            (1, 4),
        ]
    )
    print(model.summary())

    model.custom_fit(train_dataset, val_dataset, config["use_flavour_tagging"])

    return model, train_dataset.jets.scaler


def main():
    """
    Main function for the training.
    """
    args = get_parser()
    log_dir = Path(args.log_dir)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    config_loader = Configuration(args.config_file)
    config_loader.replace_with_command_line_arguments(args)
    config = config_loader.config
    if args.test:
        config["n_epochs"] = 2
        config["n_events"] = 10000
    if not args.predict:
        config_loader.dump(config["log_dir"] / "config.yaml")
        model, jet_scaler = train(config)

    else:
        model = TopographModel(config)
        jet_scaler = joblib.load(config["log_dir"] / "scaler.Jets")

    test_dataset = load_and_mask_data(
        config["test_file"],
        jets=True,
        n_events=config["n_events"],
        matchability=True,
        jet_scale=args.jet_scale,
        jet_resolution=args.jet_resolution,
    )
    test_dataset.jets.preprocess_transform(True, jet_scaler)
    model.build(
        [
            (1,) + test_dataset.jets.input_shape(config["use_flavour_tagging"])[1:],
            (1, 4),
        ]
    )
    model.load_weights(get_best_model_weights(config["log_dir"]))
    preds = model.predict(
        (
            test_dataset.jets.get_inputs(config["use_flavour_tagging"]),
            np.zeros(
                (test_dataset.jets.input_shape(config["use_flavour_tagging"])[0], 4)
            ),
        ),
        batch_size=config["batch_size"],
    )

    np.save(log_dir / "preds_tops_initial.npy", preds[0]) #1
    for i, pred in enumerate(preds[1][:-1]): # 3
        np.save(log_dir / f"preds_tops_{i}.npy", pred)
    for i, pred in enumerate(preds[2][:-1]): # 5
        np.save(log_dir / f"preds_tops_edges_{i}.npy", pred)
    np.save(log_dir / "preds_tops.npy", preds[1][-1]) #3
    np.save(log_dir / "preds_edges_tops.npy", preds[2][-1]) #5
    np.save(log_dir / "preds_tops_3.npy", preds[3]) #5
    np.save(log_dir / "preds_tops_4.npy", preds[4]) #5

    if args.all:
        test_dataset = load_and_mask_data(
            config["test_file"],
            jets=True,
            n_events=config["n_events"],
            matchability=False,
        )
        test_dataset.jets.preprocess_transform(True, jet_scaler)
        preds = model.predict(
            (
                test_dataset.jets.get_inputs(config["use_flavour_tagging"]),
                np.zeros(
                    (test_dataset.jets.input_shape(config["use_flavour_tagging"])[0], 4)
                ),
            ),
            batch_size=config["batch_size"],
        )

        np.save(log_dir / "preds_tops_initial_all.npy", preds[1])
        for i, pred in enumerate(preds[1][:-1]):
            np.save(log_dir / f"preds_tops_{i}_all.npy", pred)
        for i, pred in enumerate(preds[2][:-1]):
            np.save(log_dir / f"preds_tops_edges_{i}_all.npy", pred)
        np.save(log_dir / "preds_tops_all.npy", preds[1][-1])
        np.save(log_dir / "preds_edges_tops_all.npy", preds[2][-1])


if __name__ == "__main__":
    main()
