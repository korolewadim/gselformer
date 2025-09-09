import os
import numpy as np
import pandas as pd

import re
import random
import selfies as sf
import itertools
from functools import partial
from pathlib import Path
from rdkit import Chem
from multiprocessing import Pool

from group_selfies.group_grammar import common_r_group_replacements_grammar, _compatible
from group_selfies.utils.group_utils import extract_group_bonds


def get_actual_groups(grammar, mol):
    actual_groups = []
    mol_atomic_numbers = set([0]) | set(
        map(lambda mol: mol.GetAtomicNum(), mol.GetAtoms())
    )
    for name in list(grammar.vocab.keys()):
        group = grammar.get_group(name)
        if group_atomic_numbers[name].issubset(mol_atomic_numbers):
            for matched_atoms, matched_bonds in extract_group_bonds(group, mol, set()):
                match, _ = zip(*matched_atoms)
                if _compatible(matched_atoms, set(), group.mol, mol):
                    actual_groups.append(name)
                    break
    return actual_groups


def extract_groups(grammar, mol, selected_groups):
    mapped_atoms = set()
    groups = []
    for name in selected_groups:
        group = grammar.get_group(name)
        for matched_atoms, matched_bonds in extract_group_bonds(
            group, mol, mapped_atoms
        ):
            match, _ = zip(*matched_atoms)
            if _compatible(matched_atoms, mapped_atoms, group.mol, mol):
                groups.append((grammar.get_group(name), matched_atoms, matched_bonds))
                mapped_atoms.update(match)
    return groups


def get_diverse_group_selfies(mol, num_group_selfies=0, seed=42):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol.UpdatePropertyCache()
    init_selfies = re.sub(
        "\[|\]",
        "",
        sf.encoder(Chem.MolToSmiles(mol), strict=False)
        .replace("][", " ")
    )
    actual_groups = get_actual_groups(grammar, mol)
    selected_permutations = []

    for i in range(len(actual_groups)):
        actual_groups_copy = actual_groups.copy()
        first_el = actual_groups_copy[i]
        other_el = actual_groups_copy[:i] + actual_groups_copy[i + 1 :]
        random.Random(seed).shuffle(other_el)
        selected_permutations.append([first_el] + other_el)
    random.Random(seed).shuffle(selected_permutations)
    group_selfies = []

    for groups in selected_permutations:
        if len(set(group_selfies)) < num_group_selfies:
            group_selfies.append(
                " ".join(
                    map(
                        lambda token: re.sub("\[|\]", "", token),
                        grammar.encoder(
                            mol, extract_groups(grammar, mol, groups), join=False
                        ),
                    )
                )
            )
    diverse_group_selfies = [init_selfies] + list(set(group_selfies))
    return diverse_group_selfies


def process_smi(raw_smi, num_group_selfies=1000, seed=42):
    try:
        splitted_smi = raw_smi.split(".")

        if len(splitted_smi) == 1:
            mol = Chem.MolFromSmiles(splitted_smi[0])
            return get_diverse_group_selfies(mol, num_group_selfies, seed)
        else:
            gs_lists = []

            for smi in splitted_smi:
                mol = Chem.MolFromSmiles(smi)
                gs = get_diverse_group_selfies(mol, num_group_selfies, seed)
                gs_lists.append(gs)
                g_selfies, *dg_selfies = map(" . ".join, itertools.product(*gs_lists))
                random.Random(seed).shuffle(dg_selfies)
                dg_selfies = [g_selfies] + dg_selfies
            return dg_selfies[: num_group_selfies + 1]
    except Exception:
        return []


def preprocess_data(
    dataset_dir,
    datasplits_dir,
    task_type,
    task_name,
    split_type,
    smiles_label,
    target_label,
    num_group_selfies,
    num_processess=1,
):
    df_train = pd.read_csv(os.path.join(dataset_dir, f"ADME_{task_name}_train.csv"))
    df_test = pd.read_csv(os.path.join(dataset_dir, f"ADME_{task_name}_test.csv"))

    smiles_train, y_train = (
        df_train[smiles_label].tolist(),
        df_train[target_label].tolist(),
    )
    smiles_test, y_test = df_test[smiles_label].tolist(), df_test[target_label].tolist()

    with Pool(processes=num_processess) as pool:
        group_selfies_nested_train = pool.map(
            partial(process_smi, num_group_selfies=num_group_selfies), smiles_train
        )
    group_selfies_train = [x for xs in group_selfies_nested_train for x in xs]

    with Pool(processes=num_processess) as pool:
        group_selfies_nested_test = pool.map(
            partial(process_smi, num_group_selfies=num_group_selfies), smiles_test
        )
    group_selfies_test = [x for xs in group_selfies_nested_test for x in xs]

    data_train = [
        x
        for xs in [
            [[i, x, y] for x in x_]
            for i, (x_, y) in enumerate(zip(group_selfies_nested_train, y_train))
        ]
        for x in xs
    ]
    data_test = [
        x
        for xs in [
            [[i, x, y] for x in x_]
            for i, (x_, y) in enumerate(zip(group_selfies_nested_test, y_test))
        ]
        for x in xs
    ]

    df_train = pd.DataFrame.from_records(
        data_train, columns=["mol_index", "selfies", "label"]
    )
    df_test = pd.DataFrame.from_records(
        data_test, columns=["mol_index", "selfies", "label"]
    )

    Path(
        os.path.join(
            datasplits_dir,
            task_type,
            task_name,
            target_label.replace(" ", "_"),
            split_type,
            str(num_group_selfies),
        )
    ).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(
        os.path.join(
            datasplits_dir,
            task_type,
            task_name,
            target_label.replace(" ", "_"),
            split_type,
            str(num_group_selfies),
            "train.csv",
        ),
        index=None,
    )
    df_test.to_csv(
        os.path.join(
            datasplits_dir,
            task_type,
            task_name,
            target_label.replace(" ", "_"),
            split_type,
            str(num_group_selfies),
            "test.csv",
        ),
        index=None,
    )


grammar = common_r_group_replacements_grammar()
group_atomic_numbers = {
    k: set(map(lambda obj: obj.GetAtomicNum(), g.mol.GetAtoms()))
    for k, g in grammar.vocab.items()
}
