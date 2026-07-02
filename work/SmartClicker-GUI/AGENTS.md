# Contributor Guidance

This project is maintained in a fast-moving and sometimes very disorganized
workflow. Contributors should optimize for recoverability and clarity.

- Make small, frequent Git commits after each coherent change.
- Keep refactors separate from behavior changes whenever practical.
- Avoid leaving duplicate scripts, generated files, or one-off downloads mixed
  with source files.
- If the workspace, request, or implementation is becoming disorganized, tell
  the prompter directly and suggest one concrete change, such as pausing to
  commit, deleting duplicate files, renaming files consistently, or splitting a
  large request into smaller steps.
- If the prompter asks for work unrelated to this GUI project, suggest moving
  that unrelated topic into a separate conversation before continuing.
- Prefer clear module boundaries: serial I/O, parsing, persistence, export, and
  GUI widgets should stay easy to edit independently.
- Before finishing, run a lightweight validation command and report what passed
  or what could not be checked.
