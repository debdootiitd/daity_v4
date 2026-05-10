# notebooks/

Each phase ends with a writeup notebook (`phase_N_writeup.ipynb`) plus any
exploratory work. Notebooks should be reproducible from `make` targets and
checked-in code — never embed long inline scripts.

Conventions:
- Save outputs cleared (`nbstripout` or manual) to keep diffs reviewable.
- Use the project's logging / config rather than hard-coded paths.
