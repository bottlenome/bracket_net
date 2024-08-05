#!/usr/bin/bash -eu

echo "Running unittests"

python bracket_net/model/up_causal_unet.py

python -m bracket_net.domain.planning.reformer
python -m bracket_net.domain.planning.up_causal_unet

python -m bracket_net.domain.breakout.up_causal_unet

python -m bracket_net.domain.cube.linear

echo "PASS"
