# Contributing

## Test protection process

OLM treats tests as protected review assets. In a workflow where code and tests
may both be edited by bots or AI tools, passing tests is not enough: reviewers
also need one place to see when tests were added, removed, renamed, or changed.

The test manifest is that review surface:

- `testing/test_functions_manifest.txt` lists every discovered test function and
  a short hash of that test function body.
- `testing/scale_olm_test.py` verifies that the manifest still matches the test
  suite during pytest.
- `scripts/regenerate_manifest.py` refreshes the manifest after an intentional
  test-suite change.

When a pull request intentionally changes tests, run:

```console
python scripts/regenerate_manifest.py
```

Review the manifest diff before committing it. A new test should add one new
`hash:test_name` entry. A changed test should update the hash for that test. A
deleted or renamed test should remove or rename the corresponding entry. Treat
unexpected manifest changes as a signal to inspect the test diff carefully.

The repository also uses CODEOWNERS to require owner review for the manifest,
the manifest verifier, the regeneration script, the test workflow, the
coverage policy, the auto-merge workflow, and the CODEOWNERS file itself.
These files are the control plane for test and merge integrity.

`codecov.yml` defines the coverage policy. Project coverage uses the pull
request base as the target with a `0%` threshold, so total coverage must not go
down. Pull requests that lower project coverage need additional tests or less
untested code before their Codecov project status should pass.

```{include} ../../README.md
```
