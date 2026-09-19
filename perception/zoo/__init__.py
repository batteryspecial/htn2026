"""The model zoo. Roles, and the registry that serves them."""

from zoo.registry import (
    Entry,
    ModelUnavailableError,
    NoModelForRole,
    Registry,
    UnknownModelError,
)
from zoo.roles import ROLES, Role, types_for

__all__ = [
    "Entry", "ModelUnavailableError", "NoModelForRole", "ROLES", "Registry",
    "Role", "UnknownModelError", "types_for",
]
