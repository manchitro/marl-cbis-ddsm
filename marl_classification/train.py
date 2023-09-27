import json
from os import mkdir
from os.path import exists, isdir, join
from random import randint
import csv

import mlflow
import torch as th
import torch.nn.functional as th_fun
import torchvision.transforms as tr
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from marl_classification.data.dataset import CBISDataset

from .data import (
    KneeMRIDataset,
    MNISTDataset,
    RESISC45Dataset,
    AIDDataset,
    CBISDatasetSimplified
)
from .data import transforms as custom_tr
from .environment import (
    MultiAgent,
    obs_generic,
    trans_generic,
    detailed_episode,
    episode
)
from .infer import visualize_steps
from .metrics import ConfusionMeter, LossMeter
from .networks import ModelsWrapper
from .options import MainOptions, TrainOptions

def write_to_csv(filename, data): 
    header = ["epoch", "accuracy", "precision", "recall", "f1 score", "loss", "reward", "path", "epsilon"]

    file_exists = False
    try:
        with open(filename, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def train(
        main_options: MainOptions,
        train_options: TrainOptions
) -> None:
    assert train_options.dim == 2 or train_options.dim == 3, \
        "Only 2D or 3D is supported at the moment " \
        "for data loading and observation / transition. " \
        "See torchvision.datasets.ImageFolder"

    output_dir = train_options.output_dir

    model_dir = "models"
    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) \
            and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\""
                        f"is not a directory.")

    conf_dir = "conf_matrices"
    if not exists(join(output_dir, conf_dir)):
        mkdir(join(output_dir, conf_dir))
    if exists(join(output_dir, conf_dir)) \
            and not isdir(join(output_dir, conf_dir)):
        raise Exception(f"\"{join(output_dir, conf_dir)}\""
                        f"is not a directory.")

    steps_vis_dir = "steps_vis"
    if not exists(join(output_dir, steps_vis_dir)):
        mkdir(join(output_dir, steps_vis_dir))
    if exists(join(output_dir, steps_vis_dir)) \
            and not isdir(join(output_dir, steps_vis_dir)):
        raise Exception(f"\"{join(output_dir, steps_vis_dir)}\""
                        f"is not a directory.")

    exp_name = "MARLClassification"
    mlflow.set_experiment(exp_name)

    mlflow.start_run(run_name=f"train_{main_options.run_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    match train_options.ft_extr_str:
        case ModelsWrapper.mnist:
            dataset_constructor = MNISTDataset
        case ModelsWrapper.resisc:
            dataset_constructor = RESISC45Dataset
        case ModelsWrapper.knee_mri:
            dataset_constructor = KneeMRIDataset
        case ModelsWrapper.aid:
            dataset_constructor = AIDDataset
        case ModelsWrapper.cbis:
            dataset_constructor = CBISDataset
        case _:
            raise ValueError("Unrecognized dataset")

    nn_models = ModelsWrapper(
        train_options.ft_extr_str,
        train_options.window_size,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.hidden_size_msg,
        train_options.hidden_size_state,
        train_options.dim,
        train_options.action,
        train_options.nb_class,
        train_options.hidden_size_linear_belief,
        train_options.hidden_size_linear_action
    )

    dataset = dataset_constructor(
        train_options.resources_dir,
        img_pipeline
    )

    marl_m = MultiAgent(
        main_options.nb_agent,
        nn_models,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.window_size,
        train_options.hidden_size_msg,
        train_options.action,
        obs_generic,
        trans_generic
    )

    params = {
        "ft_extractor": train_options.ft_extr_str,
        "window_size": train_options.window_size,
        "hidden_size_belief": train_options.hidden_size_belief,
        "hidden_size_action": train_options.hidden_size_action,
        "hidden_size_msg": train_options.hidden_size_msg,
        "hidden_size_state": train_options.hidden_size_state,
        "dim": train_options.dim,
        "action": train_options.action,
        "nb_class": train_options.nb_class,
        "hidden_size_linear_belief":
            train_options.hidden_size_linear_belief,
        "hidden_size_linear_action":
            train_options.hidden_size_linear_action,
        "nb_agent": main_options.nb_agent,
        "frozen_modules": train_options.frozen_modules,
        "epsilon": train_options.epsilon,
        "epsilon_decay": train_options.epsilon_decay,
        "nb_epoch": train_options.nb_epoch,
        "learning_rate": train_options.learning_rate,
        "img_size": train_options.img_size,
        "step": main_options.step,
        "batch_size": train_options.batch_size
    }

    mlflow.log_params(params)

    json_f = open(join(output_dir, "class_to_idx.json"), "w")
    json.dump(dataset.class_to_idx, json_f)
    json_f.close()

    # save hyperparameters as json
    param_f = open(join(output_dir, "params.json"), "w")
    json.dump(params, param_f)
    param_f.close()

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    mlflow.log_param("device", device_str)

    module_to_train = ModelsWrapper.module_list \
        .difference(train_options.frozen_modules)

    # for RL agent models parameters
    optim = th.optim.Adam(
        nn_models.get_params(list(module_to_train)),
        lr=train_options.learning_rate
    )
    print('aug train dataset size: ' + str(len(dataset.aug_dataset_train)))
    print('test dataset size: ' + str(len(dataset.dataset_test)))

    # ratio_eval = 0.85
    idx = list(range(len(dataset)))
    # idx_train = idx[:int(ratio_eval * idx.size()[0])]
    # idx_test = idx[int(ratio_eval * idx.size()[0]):]
    idx_train = idx[:len(dataset.aug_dataset_train)]
    idx_test = idx[len(dataset.dataset_test):]

    train_dataset = Subset(dataset, idx_train)
    test_dataset = Subset(dataset, idx_test)
    # train_dataset = dataset.aug_dataset_train
    # test_dataset = dataset.dataset_test

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    test_dataset = DataLoader(
        test_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    epsilon = train_options.epsilon

    curr_step = 0

    conf_meter_train = ConfusionMeter(
        train_options.nb_class,
        window_size=64
    )

    path_loss_meter = LossMeter(window_size=64)
    reward_meter = LossMeter(window_size=64)
    loss_meter = LossMeter(window_size=64)

    for e in range(train_options.nb_epoch):
        nn_models.train()

        tqdm_bar = tqdm(train_dataloader)
        for x_train, y_train in tqdm_bar:
            x_train, y_train = x_train.to(th.device(device_str)), \
                               y_train.to(th.device(device_str))

            # pred = [Ns, Nb, Nc]
            # prob = [Ns, Nb]
            pred, log_proba, _ = detailed_episode(
                marl_m, x_train, epsilon,
                main_options.step,
                device_str,
                train_options.nb_class
            )

            # select last step
            last_pred = pred[-1, :, :]

            # maximize(reward) -> maximize(-error)
            reward = -th_fun.cross_entropy(
                last_pred, y_train,
                reduction="none"
            )

            # Path loss
            # sum log-probabilities (on steps), then exponential
            # maximize(probability * reward)
            path_sum = log_proba.sum(dim=0)
            path_loss = path_sum.exp() * reward.detach()

            # Losses mean on images batch
            # maximize(E[reward]) -> minimize(-E[reward])
            loss = -(path_loss + reward).mean()

            # Reset gradient
            optim.zero_grad()

            # Backward on compute graph
            loss.backward()

            # Update weights
            optim.step()

            # Update confusion meter and epoch loss sum
            conf_meter_train.add(
                last_pred.detach(),
                y_train
            )

            path_loss_meter.add(path_sum.mean().item())
            reward_meter.add(reward.mean().item())
            loss_meter.add(loss.item())

            # Compute global score
            accs, precs, recs = (
                conf_meter_train.accuracy(),
                conf_meter_train.precision(),
                conf_meter_train.recall()
            )

            if curr_step % 100 == 0:
                mlflow.log_metrics(step=curr_step, metrics={
                    "reward": reward.mean().item(),
                    "path_loss": path_sum.mean().item(),
                    "loss": loss.item(),
                    "train_prec": precs.mean().item(),
                    "train_rec": recs.mean().item(),
                    "epsilon": epsilon
                })

            tqdm_bar.set_description(
                f"Epoch {e} - Train, "
                f"acc = {format(accs.mean().item(), '.4f')}, "
                f"prec = {format(precs.mean().item(), '.4f')}, "
                f"rec = {format(recs.mean().item(), '.4f')}, "
                f"loss = {format(loss_meter.loss(), '.4f')}, "
                f"reward = {format(reward_meter.loss(), '.4f')}, "
                f"path = {format(path_loss_meter.loss(), '.4f')}, "
                f"eps = {format(epsilon, '.4f')} "
            )

            epsilon *= train_options.epsilon_decay
            epsilon = max(epsilon, 0.)

            curr_step += 1
        
        # calculate f1 score
        f1 = 2 * (precs.mean().item() * recs.mean().item()) / (precs.mean().item() + recs.mean().item())

		# append metrics to csv 
        metrics = {
            "epoch": e,
            "accuracy": format(accs.mean().item(), '.4f'),
            "precision": format(precs.mean().item(), '.4f'),
            "recall": format(recs.mean().item(), '.4f'),
            "f1 score": format(f1, '.4f'),
            "loss": format(loss_meter.loss(), '.4f'),
            "reward": format(reward_meter.loss(), '.4f'),
            "path": format(path_loss_meter.loss(), '.4f'),
            "epsilon": format(epsilon, '.4f')
        }
        write_to_csv(join(output_dir, "train_metrics.csv"), metrics)

        nn_models.eval()
        conf_meter_eval = ConfusionMeter(train_options.nb_class, None)

        with th.no_grad():
            tqdm_bar = tqdm(test_dataset)
            for x_test, y_test in tqdm_bar:
                x_test, y_test = x_test.to(th.device(device_str)), \
                                 y_test.to(th.device(device_str))

                pred, _ = episode(marl_m, x_test, 0., main_options.step)

                conf_meter_eval.add(pred.detach(), y_test)

                # Compute score
                accs, precs, recs = (
                    conf_meter_eval.accuracy(),
                    conf_meter_eval.precision(),
                    conf_meter_eval.recall()
                )

                tqdm_bar.set_description(
                    f"Epoch {e} - Eval, "
                    f"acc = {format(accs.mean().item(), '.4f')}, "
                    f"prec = {format(precs.mean().item(), '.4f')}, "
                    f"rec = {format(recs.mean().item(), '.4f')}, "
                )

        # Compute score
        accs, precs, recs = (
            conf_meter_eval.accuracy(),
            conf_meter_eval.precision(),
            conf_meter_eval.recall()
        )

        # calculate f1 score
        f1 = 2 * (precs.mean().item() * recs.mean().item()) / (precs.mean().item() + recs.mean().item())

		# append metrics to csv 
        metrics = {
            "epoch": e,
            "accuracy": format(accs.mean().item(), '.4f'),
            "precision": format(precs.mean().item(), '.4f'),
            "recall": format(recs.mean().item(), '.4f'),
            "f1 score": format(f1, '.4f'),
        }
        write_to_csv(join(output_dir, "eval_metrics.csv"), metrics)

        conf_meter_eval.save_conf_matrix_new(e, output_dir, "eval")

        mlflow.log_metrics(step=curr_step, metrics={
            "eval_acc": accs.mean().item(),
            "eval_prec": precs.mean().item(),
            "eval_recs": recs.mean().item()
        })

        nn_models.json_args(
            join(output_dir,
                 model_dir,
                 f"marl_epoch_{e}.json")
        )
        th.save(
            nn_models.state_dict(),
            join(output_dir, model_dir,
                 f"nn_models_epoch_{e}.pt")
        )

    empty_pipe = tr.Compose([
        tr.ToTensor()
    ])

    dataset_tmp = dataset_constructor(
        train_options.resources_dir,
        empty_pipe
    )

    test_dataset_ori = Subset(dataset_tmp, idx_test)
    test_dataset = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataset_ori))

    visualize_steps(
        marl_m, test_dataset[test_idx][0],
        test_dataset_ori[test_idx][0],
        main_options.step, train_options.window_size,
        output_dir, train_options.nb_class, device_str,
        dataset.class_to_idx
    )

    mlflow.end_run()
