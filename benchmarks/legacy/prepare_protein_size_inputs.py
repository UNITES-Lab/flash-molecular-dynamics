#!/usr/bin/env python
"""Prepare cross-model inputs for fixed-batch protein-size scaling benchmarks.

Run this script with the MLCG environment because decoding the serialized
CGSchNet inputs requires the local ``mlcg`` classes::

    /home/pingzhi/anaconda3/envs/mlcg/bin/python \
        submission/scripts/prepare_protein_size_inputs.py

The BioEmu benchmark times only denoising. Native BioEmu inputs retain their
real cached AlphaFold embeddings. Cross-model sequences use deterministic,
shape-equivalent embeddings tiled from the real T4 endolysin cache; this is
recorded in the manifest and keeps embedding generation outside the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCHNET = ROOT / "flash-schnet"
MACE = ROOT / "flash-mace"
BIOEMU = ROOT / "flash-bioemu"
OUT = ROOT / "submission/data/protein_size_scaling"
INPUTS = OUT / "inputs"
BIOEMU_FASTA = INPUTS / "bioemu"
BIOEMU_CACHE = OUT / "bioemu_embed_cache"
MACE_PDB = INPUTS / "mace"
CG_SHARED = INPUTS / "cgschnet_from_mace"
CG_NATIVE = INPUTS / "cgschnet_native"

CG_DATA = SCHNET / "data/simulating_a_trained_cg_model"
MACE_NATIVE = SCHNET / "third_party/torchmd-net/benchmarks/systems"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "BHA": "D",
    "CYS": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLA": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA1_TO_3 = {
    value: key
    for key, value in AA3_TO_1.items()
    if key
    in {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
}
CG_TYPE_TO_AA = {
    1: "A",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "P",
    14: "Q",
    15: "R",
    16: "S",
    17: "T",
    18: "V",
    19: "W",
    20: "Y",
}

CG_CONFIGS = {
    "opep_0034": "opep_0034_sim_epoch_73.yaml",
    "opep_0015": "opep_0015_sim_epoch_73.yaml",
    "cln": "cln_sim_epoch_73-dev.yaml",
    "2jof": "2jof_sim_epoch_73-dev.yaml",
    "ext_PUMA": "ext_PUMA_alone_sim_epoch_73.yaml",
    "1fme": "1fme_sim_epoch_73-dev.yaml",
    "1yrf": "1yrf_sim_epoch_73-dev.yaml",
    "1enh": "1enh_sim_epoch_73-dev.yaml",
    "2nuz": "2nuz_sim_epoch_73.yaml",
    "2ci2": "2ci2_sim_epoch_73.yaml",
    "3zbe": "3zbe_sim_langevin_epoch_73.yaml",
    "2a3d": "2a3d_sim_epoch_73-dev.yaml",
    "1ris": "1ris_sim_epoch_73.yaml",
    "1d3z_PUMA": "1d3z_with_ext_PUMA_sim_epoch_73-dev.yaml",
    "mcl1_PUMA": "mcl1_with_ext_PUMA_sim_epoch_73.yaml",
}
CG_STRUCTURES = {
    "opep_0034": "opep_0034_configurations.pt",
    "opep_0015": "opep_0015_configurations.pt",
    "cln": "cln_configurations_single.pt",
    "2jof": "2jof_configurations_single.pt",
    "ext_PUMA": "ext_PUMA_alone_configurations.pt",
    "1fme": "1fme_configurations.pt",
    "1yrf": "1yrf_configurations_single.pt",
    "1enh": "1enh_configurations_single.pt",
    "2nuz": "2nuz_configurations.pt",
    "2ci2": "2ci2_configurations.pt",
    "3zbe": "3zbe_configurations.pt",
    "2a3d": "2a3d_configurations_single.pt",
    "1ris": "1ris_configurations.pt",
    "1d3z_PUMA": "1d3z_with_ext_PUMA_configurations.pt",
    "mcl1_PUMA": "mcl1_with_ext_PUMA_configurations.pt",
}

MACE_NATIVE_PROTEINS = {
    "chignolin": "chignolin.pdb",
    "dhfr": "dhfr.pdb",
    "factorIX": "factorIX.pdb",
}

BIOEMU_NATIVE = {
    "HHH_rd1_0335": BIOEMU / "dev/benchmark_inputs/hhh_rd1_0335.a3m",
    "ubiquitin": BIOEMU / "dev/benchmark_inputs/ubiquitin_human_76.a3m",
    "T4_endolysin": BIOEMU / "dev/benchmark_inputs/t4_endolysin_p00720.a3m",
}

RCSB_FROM_SCHNET = {
    "2jof": "2JOF",
    "1fme": "1FME",
    "1yrf": "1YRF",
    "1enh": "1ENH",
    "2nuz": "2NUZ",
    "2ci2": "2CI2",
    "3zbe": "3ZBE",
    "2a3d": "2A3D",
    "1ris": "1RIS",
}


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def first_fasta_sequence(path: Path) -> str:
    sequence: list[str] = []
    found = False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if found:
                break
            found = True
            continue
        if found:
            sequence.append("".join(c for c in line.strip() if c.isupper()))
    return "".join(sequence)


def pdb_chains(path: Path) -> dict[str, dict[str, object]]:
    residues: dict[str, list[tuple[tuple[str, str], str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    atoms: defaultdict[str, int] = defaultdict(int)
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21:22]
        residue_key = (line[22:26], line[26:27])
        residue_name = line[17:20].strip()
        atoms[chain] += 1
        full_key = (chain, *residue_key)
        if full_key not in seen and residue_name in AA3_TO_1:
            seen.add(full_key)
            residues[chain].append((residue_key, AA3_TO_1[residue_name]))
    return {
        chain: {
            "sequence": "".join(aa for _, aa in chain_residues),
            "residue_keys": [list(key) for key, _ in chain_residues],
            "atoms": atoms[chain],
        }
        for chain, chain_residues in residues.items()
    }


def normalize_pdb(
    source: Path,
    destination: Path,
    *,
    chain: str | None = None,
) -> None:
    lines: list[str] = []
    serial = 1
    for line in source.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if chain is not None and line[21:22] != chain:
            continue
        residue_name = line[17:20].strip()
        if residue_name not in AA3_TO_1:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        canonical = AA1_TO_3[AA3_TO_1[residue_name]]
        normalized = "ATOM  " + f"{serial:5d}" + line[11:16] + " " + f"{canonical:>3s}" + line[20:]
        lines.append(normalized[:80].ljust(80))
        serial += 1
    lines.extend(["TER", "END"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def count_pdb_atoms(path: Path) -> int:
    return sum(
        line.startswith(("ATOM  ", "HETATM"))
        for line in path.read_text(errors="ignore").splitlines()
    )


def decode_cg_input(path: Path) -> tuple[int, str]:
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    frame = loaded[0] if isinstance(loaded, list) else loaded
    atom_types = frame.atom_types.detach().cpu().reshape(-1).tolist()
    sequence = "".join(CG_TYPE_TO_AA[int(value)] for value in atom_types if int(value) <= 20)
    return int(frame.pos.shape[0]), sequence


def fetch_rcsb(pdb_id: str) -> Path:
    raw_dir = MACE_PDB / "rcsb_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{pdb_id.lower()}.pdb"
    if not destination.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    return destination


def matching_chain(path: Path, target: str) -> tuple[str, float, str]:
    chains = pdb_chains(path)
    if not chains:
        raise ValueError(f"No protein chains found in {path}")
    scores = {
        chain: SequenceMatcher(None, target, str(data["sequence"])).ratio()
        for chain, data in chains.items()
    }
    chain = max(scores, key=scores.get)
    return chain, scores[chain], str(chains[chain]["sequence"])


def write_fasta(path: Path, name: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = "\n".join(sequence[i : i + 80] for i in range(0, len(sequence), 80))
    path.write_text(f">{name}\n{wrapped}\n")


def copy_native_embedding(sequence: str) -> bool:
    digest = sequence_hash(sequence)
    source_dir = BIOEMU / "profiles/benchmark_embed_cache"
    copied = False
    for suffix in ("_single.npy", "_pair.npy", ".fasta"):
        source = source_dir / f"{digest}{suffix}"
        if source.exists():
            shutil.copy2(source, BIOEMU_CACHE / source.name)
            copied = True
    return copied


def create_shape_equivalent_embedding(
    sequence: str,
    source_single: np.ndarray,
    source_pair: np.ndarray,
) -> None:
    digest = sequence_hash(sequence)
    n = len(sequence)
    indices = np.arange(n) % source_single.shape[0]
    single = source_single[indices].copy()
    pair = source_pair[indices[:, None], indices[None, :]].copy()
    np.save(BIOEMU_CACHE / f"{digest}_single.npy", single)
    np.save(BIOEMU_CACHE / f"{digest}_pair.npy", pair)
    write_fasta(BIOEMU_CACHE / f"{digest}.fasta", digest, sequence)


def write_cg_benchmark_config(path: Path, structure: Path, name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "simulation:",
                "  friction: 1.0",
                "  n_timesteps: 200",
                "  dt: 0.0",
                "  save_forces: false",
                "  save_energies: false",
                "  save_interval: 40",
                "  device: cuda",
                "  dtype: single",
                "  export_interval: 200",
                "  log_interval: 40",
                "  log_type: write",
                "  specialize_priors: true",
                "  create_checkpoints: false",
                "  read_checkpoint_file: false",
                f"  filename: protein_size_{name}",
                "betas:",
                "- 1.677399001146518",
                f"model_file: {(CG_DATA / 'model_and_prior.pt').resolve()}",
                f"structure_file: {structure.resolve()}",
                "",
            ]
        )
    )


def convert_mace_inputs_to_cg(
    mace_native: list[dict[str, object]],
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    prior_config = (
        SCHNET
        / "third_party/mlcg-tk/examples/configuration_files/transferable_priors.yaml"
    )
    generator = SCHNET / "third_party/mlcg-tk/src/mlcg_tk/scripts/gen_sim_input.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(SCHNET / "src"),
            str(SCHNET / "third_party/mlcg-tk/src"),
            env.get("PYTHONPATH", ""),
        ]
    )

    for item in mace_native:
        name = str(item["id"])
        source = ROOT / str(item["pdb"])
        system_dir = CG_SHARED / name
        system_dir.mkdir(parents=True, exist_ok=True)
        input_config = system_dir / "conversion.yaml"
        input_config.write_text(
            "\n".join(
                [
                    f"dataset_name: {name}",
                    f"raw_data_dir: {source.parent.resolve()}/",
                    f"save_dir: {system_dir.resolve()}/",
                    "tag: from_mace",
                    "pdb_fns:",
                    f"- {source.name}",
                    "cg_atoms:",
                    "- N",
                    "- CA",
                    "- CB",
                    "- C",
                    "- O",
                    "embedding_map: mlcg_tk.input_generator.CGEmbeddingMapFiveBead",
                    "embedding_func: mlcg_tk.input_generator.embedding_fivebead",
                    "skip_residues: []",
                    "copies: 1",
                    "",
                ]
            )
        )
        structure = system_dir / f"{name}_from_mace_configurations.pt"
        if not structure.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "process_sim_input",
                    "--config",
                    str(input_config),
                    "--config",
                    str(prior_config),
                ],
                cwd=SCHNET,
                env=env,
                check=True,
            )
        beads, sequence = decode_cg_input(structure)
        benchmark_config = system_dir / "benchmark.yaml"
        write_cg_benchmark_config(benchmark_config, structure, name)
        converted.append(
            {
                "id": name,
                "beads": beads,
                "residues": len(sequence),
                "sequence": sequence,
                "structure": str(structure.relative_to(ROOT)),
                "config": str(benchmark_config.relative_to(ROOT)),
                "source": "flash-mace-to-schnet",
            }
        )
    return converted


def main() -> None:
    for directory in (
        OUT,
        INPUTS,
        BIOEMU_FASTA,
        BIOEMU_CACHE,
        MACE_PDB,
        CG_SHARED,
        CG_NATIVE,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    schnet_systems: list[dict[str, object]] = []
    schnet_sequences: dict[str, str] = {}
    for name, structure_name in CG_STRUCTURES.items():
        structure = CG_DATA / structure_name
        beads, sequence = decode_cg_input(structure)
        schnet_sequences[name] = sequence
        benchmark_dir = CG_NATIVE / name
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_config = benchmark_dir / "benchmark.yaml"
        write_cg_benchmark_config(benchmark_config, structure, name)
        schnet_systems.append(
            {
                "id": name,
                "beads": beads,
                "residues": len(sequence),
                "sequence": sequence,
                "structure": str(structure.relative_to(ROOT)),
                "config": str(benchmark_config.relative_to(ROOT)),
                "source_config": str((CG_DATA / CG_CONFIGS[name]).relative_to(ROOT)),
                "source": "flash-schnet",
            }
        )

    mace_native: list[dict[str, object]] = []
    mace_sequences: dict[str, str] = {}
    for name, pdb_name in MACE_NATIVE_PROTEINS.items():
        source = MACE_NATIVE / pdb_name
        destination = MACE_PDB / f"{name}_native.pdb"
        normalize_pdb(source, destination)
        chains = pdb_chains(destination)
        sequence = max((str(data["sequence"]) for data in chains.values()), key=len)
        mace_sequences[name] = sequence
        mace_native.append(
            {
                "id": name,
                "atoms": count_pdb_atoms(destination),
                "residues": len(sequence),
                "sequence": sequence,
                "pdb": str(destination.relative_to(ROOT)),
                "source": "flash-mace-native",
            }
        )

    schnet_shared = convert_mace_inputs_to_cg(mace_native)

    mace_shared: list[dict[str, object]] = []
    for name, pdb_id in RCSB_FROM_SCHNET.items():
        raw = fetch_rcsb(pdb_id)
        chain, match, chain_sequence = matching_chain(raw, schnet_sequences[name])
        destination = MACE_PDB / f"{name}_from_schnet.pdb"
        normalize_pdb(raw, destination, chain=chain)
        mace_shared.append(
            {
                "id": name,
                "atoms": count_pdb_atoms(destination),
                "residues": len(chain_sequence),
                "sequence": chain_sequence,
                "target_sequence": schnet_sequences[name],
                "sequence_match": match,
                "pdb": str(destination.relative_to(ROOT)),
                "rcsb_id": pdb_id,
                "rcsb_chain": chain.strip() or "blank",
                "source": "flash-schnet-to-mace",
            }
        )

    context_sources: defaultdict[str, set[str]] = defaultdict(set)
    context_names: defaultdict[str, list[str]] = defaultdict(list)
    native_sequences: dict[str, str] = {}
    for name, path in BIOEMU_NATIVE.items():
        sequence = first_fasta_sequence(path)
        native_sequences[name] = sequence
        context_sources[sequence].add("flash-bioemu-native")
        context_names[sequence].append(name)
    for name, sequence in schnet_sequences.items():
        if name in {"1d3z_PUMA", "mcl1_PUMA"}:
            continue
        context_sources[sequence].add("flash-schnet")
        context_names[sequence].append(name)
    for name, sequence in mace_sequences.items():
        context_sources[sequence].add("flash-mace")
        context_names[sequence].append(name)

    t4_sequence = native_sequences["T4_endolysin"]
    t4_digest = sequence_hash(t4_sequence)
    source_cache = BIOEMU / "profiles/benchmark_embed_cache"
    source_single = np.load(source_cache / f"{t4_digest}_single.npy")
    source_pair = np.load(source_cache / f"{t4_digest}_pair.npy")

    bioemu_contexts: list[dict[str, object]] = []
    for sequence in sorted(context_sources, key=lambda value: (len(value), value)):
        name = context_names[sequence][0]
        fasta = BIOEMU_FASTA / f"{name}_{len(sequence)}aa.fasta"
        write_fasta(fasta, name, sequence)
        actual = copy_native_embedding(sequence)
        if not actual:
            create_shape_equivalent_embedding(sequence, source_single, source_pair)
        bioemu_contexts.append(
            {
                "id": name,
                "aliases": context_names[sequence],
                "residues": len(sequence),
                "sequence": sequence,
                "fasta": str(fasta.relative_to(ROOT)),
                "embedding": "native_cached" if actual else "shape_equivalent_tiled_t4",
                "sources": sorted(context_sources[sequence]),
            }
        )

    manifest = {
        "notes": {
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
            "bioemu_embedding_policy": (
                "Native BioEmu inputs use their cached AlphaFold embeddings. Cross-model "
                "sequences use deterministic T4-tiled embeddings with the exact target shapes; "
                "these inputs are for execution scaling only, not biological sample analysis."
            ),
            "mace_shared_policy": (
                "RCSB heavy-atom chains are selected by sequence similarity to the SchNet input "
                "and normalized to the 20 canonical residue names."
            ),
        },
        "schnet_native": sorted(schnet_systems, key=lambda item: int(item["beads"])),
        "schnet_shared_from_mace": sorted(
            schnet_shared, key=lambda item: int(item["beads"])
        ),
        "mace_native": sorted(mace_native, key=lambda item: int(item["atoms"])),
        "mace_shared_from_schnet": sorted(
            mace_shared, key=lambda item: int(item["atoms"])
        ),
        "bioemu_contexts": bioemu_contexts,
        "bioemu_embed_cache": str(BIOEMU_CACHE.relative_to(ROOT)),
    }
    manifest_path = OUT / "protein_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    print(
        f"SchNet systems: {len(schnet_systems)}; "
        f"SchNet shared: {len(schnet_shared)}; "
        f"MACE native/shared: {len(mace_native)}/{len(mace_shared)}; "
        f"BioEmu contexts: {len(bioemu_contexts)}"
    )


if __name__ == "__main__":
    main()
