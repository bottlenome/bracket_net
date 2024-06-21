#!/usr/bin/bash -eu

echo "Running unittests"

python -m bracket_net.domain.planning.reformer
python -m bracket_net.domain.planning.up_causal_unet

echo "PASS"