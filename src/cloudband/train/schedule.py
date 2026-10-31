"""One training cell for every session: take whatever work is free, until none is left.

Every account runs the notebook from top to bottom, and this decides what it trains:
first the learning-rate candidates of every network, then the full runs, one seed at a
time across the networks, so the seeds that the comparison pairs are finished first.
A unit another session holds is left to it. When the only work left is the full runs
of a network whose search another session is still finishing, the session waits and
looks again; when everything left is being trained elsewhere, it stops.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastai.data.core import DataLoaders

from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import SearchBusy, search_learning_rate
from cloudband.train.phase2 import full_protocol, train_if_missing
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    PROXY_FROZEN_EPOCHS,
    PROXY_UNFROZEN_EPOCHS,
    LrSearchManifest,
    TrainProtocol,
)
from cloudband.train.store import RunStore


@dataclass(frozen=True)
class Network:
    """One model trained on its own: an OCM backbone, or Swin+UPerNet."""

    label: str
    search_protocol: TrainProtocol
    base_protocol: TrainProtocol
    builder: Callable
    dls: DataLoaders
    suffix: str = ""


def phase2_networks(ocm_protocol, swin_protocol, ocm_dls, swin_dls, pretrained=True):
    """The three networks of the comparison, in the order their work is taken."""
    from cloudband.models.ocm import PAPER_BACKBONES, build_unet, ocm_search_protocol
    from cloudband.models.swin_upernet import build_swin_upernet

    networks = [
        Network(
            label=f"OCM {backbone}",
            search_protocol=ocm_search_protocol(ocm_protocol, backbone),
            base_protocol=ocm_protocol,
            builder=lambda backbone=backbone: build_unet(backbone, pretrained=pretrained),
            dls=ocm_dls,
            suffix=backbone,
        )
        for backbone in PAPER_BACKBONES
    ]
    networks.append(
        Network(
            label="Swin+UPerNet",
            search_protocol=swin_protocol,
            base_protocol=swin_protocol,
            builder=lambda: build_swin_upernet(pretrained=pretrained),
            dls=swin_dls,
        )
    )
    return networks


@dataclass
class Outcome:
    trained: list[str] = field(default_factory=list)
    held_elsewhere: list[str] = field(default_factory=list)
    waiting_for_search: list[str] = field(default_factory=list)
    at_grid_edge: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return not (self.held_elsewhere or self.waiting_for_search or self.at_grid_edge)


class Scheduler:
    def __init__(
        self,
        networks: list[Network],
        store: RunStore,
        seeds: tuple,
        grid: tuple = LR_SEARCH_GRID,
        search_seed: int | None = None,
        break_locks: bool = False,
        say: Callable = print,
    ):
        self.networks = list(networks)
        self.store = store
        self.seeds = tuple(seeds)
        self.grid = tuple(grid)
        self.search_seed = self.seeds[0] if search_seed is None else search_seed
        self.break_locks = break_locks
        self.say = say
        self.budget = (PROXY_FROZEN_EPOCHS, PROXY_UNFROZEN_EPOCHS)

    def _candidate_done(self, network: Network, rate: float) -> bool:
        record = self.store.load_search_run(
            network.search_protocol, rate, self.budget, self.search_seed
        )
        return record is not None

    def search_of(self, network: Network) -> LrSearchManifest | None:
        """The finished search of a network, saved once complete; None while it is not."""
        saved = self.store.load_search(network.search_protocol)
        if saved is not None and tuple(saved.grid) == self.grid:
            return saved
        if not all(self._candidate_done(network, rate) for rate in self.grid):
            return None
        search = search_learning_rate(
            network.dls,
            network.search_protocol,
            seed=self.search_seed,
            model_builder=network.builder,
            grid=self.grid,
            store=self.store,
        )
        self.store.save_search(network.search_protocol, search)
        return search

    def _run_name(self, network: Network, search: LrSearchManifest, seed: int) -> str:
        protocol = full_protocol(network.base_protocol, search.winner().learning_rate)
        return checkpoint_name(protocol, seed, network.suffix)

    def step(self) -> str | None:
        """Train the first free unit, if any. Returns its name, or None."""
        for rate in self.grid:
            for network in self.networks:
                if self._candidate_done(network, rate):
                    continue
                try:
                    self.say(f"{network.label}: learning-rate candidate {rate:g}")
                    search_learning_rate(
                        network.dls,
                        network.search_protocol,
                        seed=self.search_seed,
                        model_builder=network.builder,
                        grid=(rate,),
                        store=self.store,
                        break_locks=self.break_locks,
                    )
                except SearchBusy:
                    self.say("   held by another session")
                    continue
                return f"{network.label} candidate {rate:g}"

        for seed in self.seeds:
            for network in self.networks:
                search = self.search_of(network)
                if search is None or search.winner_at_edge():
                    continue
                protocol = full_protocol(network.base_protocol, search.winner().learning_rate)
                name = checkpoint_name(protocol, seed, network.suffix)
                if self.store.is_done(name):
                    continue
                self.say(f"{name}: learning rate {protocol.learning_rate:g}")
                status = train_if_missing(
                    network.dls,
                    protocol,
                    search,
                    seed,
                    network.builder,
                    self.store,
                    checkpoint_suffix=network.suffix,
                    break_locks=self.break_locks,
                )
                if status == "trained":
                    return name
                self.say(f"   {status}")
        return None

    def remaining(self) -> Outcome:
        """What is left once nothing can be taken: held, waiting, or stopped at the edge."""
        outcome = Outcome()
        for network in self.networks:
            search = self.search_of(network)
            if search is None:
                missing = [r for r in self.grid if not self._candidate_done(network, r)]
                outcome.held_elsewhere += [
                    f"{network.label} candidate {rate:g}" for rate in missing
                ]
                outcome.waiting_for_search += [
                    f"{network.label} seed {seed}" for seed in self.seeds
                ]
                continue
            if search.winner_at_edge():
                outcome.at_grid_edge.append(
                    f"{network.label}: winner {search.winner().learning_rate:g} is on an "
                    f"end of the grid {self.grid}"
                )
                continue
            for seed in self.seeds:
                name = self._run_name(network, search, seed)
                if not self.store.is_done(name):
                    outcome.held_elsewhere.append(name)
        return outcome


def run_everything(
    networks: list[Network],
    store: RunStore,
    seeds: tuple,
    grid: tuple = LR_SEARCH_GRID,
    break_locks: bool = False,
    wait: bool = True,
    poll_seconds: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable = print,
) -> Outcome:
    """Train every free unit of the comparison, one after another, then report.

    With `wait`, a session that runs out of free work while some network's full runs
    still wait for its search looks again every `poll_seconds`, since those runs open
    up as soon as the search ends (or its holder's lock expires). When all that is left
    is being trained by other sessions, it returns.
    """
    store.wake()
    scheduler = Scheduler(networks, store, seeds, grid=grid, break_locks=break_locks, say=say)
    trained: list[str] = []
    while True:
        done = scheduler.step()
        if done is not None:
            trained.append(done)
            say(f"finished: {done}")
            continue
        outcome = scheduler.remaining()
        outcome.trained = trained
        for note in outcome.at_grid_edge:
            say(f"STOPPED, the grid must be extended: {note}")
        if outcome.finished:
            say("every search and every run of the comparison is done")
            return outcome
        if outcome.waiting_for_search and wait:
            say(
                f"nothing free right now: {len(outcome.held_elsewhere)} unit(s) held by "
                f"other sessions, {len(outcome.waiting_for_search)} run(s) waiting for a "
                f"search; looking again in {poll_seconds / 60:.0f} min"
            )
            sleep(poll_seconds)
            store.wake()
            continue
        if outcome.held_elsewhere:
            say(
                "nothing left for this session: "
                f"{len(outcome.held_elsewhere)} unit(s) are being trained by other sessions"
            )
        return outcome
