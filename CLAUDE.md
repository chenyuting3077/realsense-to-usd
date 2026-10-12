# realsense-to-usd

## Writing READMEs, docs and comments

- Describe the code as it is now. Do not write about past changes ("before", "used then", "now X instead of Y",
  "was N ms", "changed from ...") in READMEs, docs or code comments unless the user asks for it.
- A comment or docstring gives what the code does and, where it is not obvious, why. Not how it got there:
  that belongs in the commit message.
- Benchmark and comparison docs (docs/<topic>/) record the experiments the user asked for; their before / after
  tables are the content and stay.

## Where files go

- docs/ holds documents only (README.md / README.zh-TW.md, images). No code there: benchmark and evaluation
  scripts go to bench/ (run as `python -m bench.<name>` or `bench/<name>.sh`), system config files to system/.
