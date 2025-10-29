# Trending Respositories Investigation

This repository will analyse the top N (N=100 at the moment) trending models on the Hugging Face Hub.
The purpose of this is to check for `library name`, and the `pipeline` tag, which help with code snippet
population on the model page. We want to alert the 🥑 team of any metadata issues.

The next goal is to execute all the code snippets from the top N models via [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).
This is done on a daily basis, and the team would be alerted for any issues while execution.

A nit about HF Jobs:
We access the code snippets from the models and upload the snippet as a `.py` file to a Hugging Face Dataset.
This is done in order to use the script via a URL. We also use `uv` scripts to make the script runnable on its own.


Some edge cases to note:
1. The repository might have no discussion tab (no need to engage)
2. The repository may be gated (needs to be accessed)

The artifacts generated:
You will find a list of datasets in the `configuration.py` file. Those datasets
are generated as artifacts when you run the repository.

Slack Messages:
1. Alert on metadata issues.
2. Also add pending PRs from the team.
3. Problems with the code execution.
