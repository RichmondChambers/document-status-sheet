"""Nightly entrypoint to refresh the Drive-backed index without cooldown."""

from index_builder import sync_drive_and_rebuild_index_if_needed


if __name__ == "__main__":
    rebuilt = sync_drive_and_rebuild_index_if_needed(
        bypass_cooldown=True,
        respect_overnight_window=False,
    )
    status = "rebuilt" if rebuilt else "skipped (no changes)"
    print(f"Nightly Drive index refresh {status}.")
