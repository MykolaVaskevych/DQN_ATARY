# A DQN implmentation for the Atari Breakdown game

## Quick Start

```bash
uv sync
```

```bash
uv run marimo edit notebook/CS4287-Assignment-2-Deep-Reinforcment-Learning.py
```

## Dependencies

- uv (python package manager)
- python3

### Dependency Installation

Install uv with our standalone installers or your package manager of choice.

#### Standalone installer

uv provides a standalone installer to download and install uv:

=== "macOS and Linux"

    Use `curl` to download the script and execute it with `sh`:

    ```console
    $ curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    If your system doesn't have `curl`, you can use `wget`:

    ```console
    $ wget -qO- https://astral.sh/uv/install.sh | sh
    ```

    Request a specific version by including it in the URL:

    ```console
    $ curl -LsSf https://astral.sh/uv/0.9.14/install.sh | sh
    ```

=== "Windows"

    Use `irm` to download the script and execute it with `iex`:

    ```pwsh-session
    PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Changing the [execution policy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4#powershell-execution-policies) allows running a script from the internet.

    Request a specific version by including it in the URL:

    ```pwsh-session
    PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.9.14/install.ps1 | iex"
    ```

