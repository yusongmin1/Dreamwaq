"""Ensure local repo packages win over conflicting site-packages stubs."""
import os
import sys


def setup_repo_paths():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    rsl_rl_root = os.path.join(repo_root, "rsl_rl")
    if rsl_rl_root in sys.path:
        sys.path.remove(rsl_rl_root)
    sys.path.insert(0, rsl_rl_root)

    for module_name, marker in (("legged_gym", "/site-packages/legged_gym/"),
                                ("rsl_rl", "/site-packages/rsl_rl/")):
        existing = sys.modules.get(module_name)
        if existing is None:
            continue
        existing_path = getattr(existing, "__file__", "") or ""
        if marker in existing_path:
            del sys.modules[module_name]
