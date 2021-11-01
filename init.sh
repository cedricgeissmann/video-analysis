#! /usr/bin/env bash

VENV=venv
SESSION_NAME=video-analysis

echo "Check if venv exists..."
if [ -d ${VENV} ]; then
    echo "The venv exists"
else
    echo "Creating new venv"
    python3 -m venv ${VENV}
    echo "New environment ${VENV} created."
    echo "Check if there is anything to install..."
    if [ -e requirements.txt ]; then
        source ${VENV}/bin/activate
        pip install -r requirements.txt
        deactivate
        echo "Installation is done..."
    fi
fi

tmux new-session -d -s ${SESSION_NAME}
tmux send-keys "source ${VENV}/bin/activate" Enter
tmux send-keys "ls -l" Enter

tmux attach -t ${SESSION_NAME}

