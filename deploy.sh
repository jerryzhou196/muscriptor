#!/usr/bin/env bash
#

set -ex

REMOTE="ssh://root@muscriptor.kyutai.org"

: "${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token before deploying}"

# Kyutai's own GA property — only this branch deploys muscriptor.kyutai.org.
export VITE_GA_MEASUREMENT_ID=G-FNB2XC72R7

docker -H "${REMOTE}" compose -f swarm.yml build --push

docker -H "${REMOTE}" stack deploy \
    --with-registry-auth \
    -c swarm.yml muscriptor

