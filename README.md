# Git Question-Answering Chat Bot

This chat bot is built on top of LangChain and uses the [Pro Git
book](https://git-scm.com/book/en/v2) as documentation.

This is a chatbot about Git where the training pipeline was built using DVC.

It was forked off from a generic [LangChain Demo](https://github.com/hwchase17/notion-qa).

# Environment Setup

First you need to do a git pull of the code:
```shell
git clone git@github.com:iterative/llm-demo.git
cd llm-demo
```

In order to set your environment up to run the code here, first install all requirements in a virtual env:
```shell
virtualenv env --python=python3.9
source env/bin/activate
pip install -r requirements.txt
```

Then set your Hugging Face API key (if you don't have one, get one
[here](https://huggingface.co/docs/hub/en/security-tokens)):
```shell
  export HUGGINGFACEHUB_API_TOKEN=....
```
The preceeding spaces prevent the API key from staying in your bash history if that is [configured](https://stackoverflow.com/questions/6475524/how-do-i-prevent-commands-from-showing-up-in-bash-history).

# Running

Now you should be ready to re-run the training pipeline. Assuming you have not changed anything, nothing should need to run. Everything can be re-used for the DVC pull:
```shell
dvc repro --pull
```

Now you can startup the web UI using:
```shell
streamlit run main.py
```
The log of interactions can be found in `chat.log`.
