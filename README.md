# Trending Respositories Investigation

This repository will analyse the top 100 trending models on the Hugging Face Hub.
The purpose of this is to check for `library name`, and the `pipeline` tag. These
metadata is useful for repositories as we can populate code snippets with them.

Some edge cases to note:
- The repository may be gated (needs to be accessed)
- The repository might have no discussion tab (no need to engage)

We also need to run all the code for each repository to:
- Make sure the first point of contact for users is running

The artifacts generated:
- A running dataset for all the trending models everyday
- Models which do not have model name
- Models which do not have pipeline tag
- Models which were not able to run (this is taken care of by HF Jobs)

A nit about HF Jobs:
- We need to run the model on the Jobs API
- We have to create the code snippet (as a `.py` file) and host it inside a dataset
- The Jobs API takes the `.py` file and then runs it.

Ideas:
- The python script can have a call to the slack api, so that it sends a report
for the avocado team to view.
- We also need to track avocado team members comments, and check whether it has completed (this is worth checking)
- Do we need custom code tracking? Maybe useful to see (but this is just a report for transformers team to check)

Some Slack Caveats:
- Need to understand the slack restriction for characters
- Sometimes there might be no issues (the day we all strive to see)

