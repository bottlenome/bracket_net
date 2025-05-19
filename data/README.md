# Preparation
## planning-datasets
```
git submodule update --init --recursive
```
## cube dataset
`R222ShortestAll.pkl` is automatically downloaded from Google Drive if it is not
found locally.  The file can also be fetched manually with:

```bash
wget -O data/R222ShortestAll.pkl \
  "https://drive.google.com/uc?export=download&id=1H42CfagdAVYuYDb9RDycPFZ3DEW6U_4k"
```
## Breakout dataset
Install gsutil and
```
gsutil -m cp gs://atari-replay-datasets/dqn/Breakout .
```