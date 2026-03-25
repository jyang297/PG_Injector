from __future__ import annotations

# Goal:
# Write a regression that fails when a key column disappears
# from the final metadata bundle.
#
# Suggested locked-in behavior:
# - query: "blocked by legal or waiting for exec approval"
# - expected columns: {"compliance_posture", "contract_state"}


def main():
    # TODO:
    # 1. Build retrieval inputs
    # 2. Build the query embedding
    # 3. Fetch hybrid hits
    # 4. Roll them up into candidate columns
    # 5. Assert that both expected columns are present
    #
    # Keep the test small and direct.
    raise NotImplementedError


if __name__ == "__main__":
    main()
