import random
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class ActionInstance:
    name: str
    method: str
    path: str
    json: Dict


@dataclass
class ActionTemplate:
    """Defines a legal action and how to sample parameters."""

    name: str
    method: str
    path: str
    sampler: Callable[[random.Random], Dict]

    def sample(self, rng: random.Random) -> ActionInstance:
        return ActionInstance(
            name=self.name,
            method=self.method,
            path=self.path,
            json=self.sampler(rng),
        )


class ActionSpace:
    """Controlled action space to avoid unsafe or out-of-scope traffic."""

    def __init__(self) -> None:
        inventory_items = ["widgets", "gadgets", "doodads"]
        self.templates: List[ActionTemplate] = [
            ActionTemplate(
                name="reset",
                method="POST",
                path="/reset",
                sampler=lambda rng: {},
            ),
            ActionTemplate(
                name="restock",
                method="POST",
                path="/inventory",
                sampler=lambda rng: {
                    "item": rng.choice(inventory_items),
                    "quantity": rng.randint(1, 5),
                },
            ),
            ActionTemplate(
                name="drain_inventory",
                method="POST",
                path="/inventory",
                sampler=lambda rng: {
                    "item": rng.choice(inventory_items),
                    # Negative restock lets the engine probe edge cases without exploits.
                    "quantity": -rng.randint(1, 3),
                },
            ),
            ActionTemplate(
                name="purchase",
                method="POST",
                path="/purchase",
                sampler=lambda rng: {
                    "item": rng.choice(inventory_items),
                    "quantity": rng.randint(1, 6),
                    "expedite": rng.random() < 0.4,
                },
            ),
            ActionTemplate(
                name="toggle_mode",
                method="POST",
                path="/mode",
                sampler=lambda rng: {"mode": rng.choice(["normal", "maintenance", "slow"])},
            ),
        ]

    def all_templates(self) -> List[ActionTemplate]:
        return list(self.templates)

    def sample(self, rng: random.Random) -> ActionInstance:
        tmpl = rng.choice(self.templates)
        return tmpl.sample(rng)
