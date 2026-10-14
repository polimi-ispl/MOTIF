# DiCaprio demo

Enrolls Leonardo DiCaprio from eight genuine reference clips, then scores three
genuine test clips and four deepfakes. The demo reports the global, mouth and
fused MOTIF scores, ranking AUC and the gap between the worst genuine score and
the best fake score.

Start with the [MOTIF prerequisites](../README.md#install): Python 3.10 or newer
and an NVIDIA GPU are required. The demo additionally needs **Bash, yt-dlp and
the FFmpeg executable with libx264**. Run from the repository root:

```bash
python -m pip install -e '.[demo]'
cd demo
./download.sh
./prepare_clips.sh
python quickstart.py
```

Downloads use Firefox cookies by default. Set `COOKIES_FROM_BROWSER=chrome` for
Chrome or `COOKIES_FROM_BROWSER=none` for anonymous access. YouTube and TikTok
may block downloads; the source URLs are preserved in [links.txt](links.txt).

Already have the prepared clips?

```bash
python quickstart.py --clips path/to/clips --out results --device cuda:0
```

Use `--batch-size N` to reduce GPU memory use. Coefficient extraction is
batch-sensitive, so changing it can slightly change the scores; the recorded
result uses the default of 8. The runner otherwise fixes the cohort and scoring
procedure: eight references, three genuine queries and four fake queries, with
every decoded frame processed by the released MOTIF models.

## Output

The demo writes only **`demo/results/scores.json`** by default. It contains all
seven global, mouth and fused scores, the AUC, genuine/fake margin, reference
clip names, MOTIF version, device and batch size. Downloaded videos, prepared
clips and results are ignored by Git and remain under `demo/`.

Higher MOTIF scores mean greater consistency with the enrolled person. Scores
are not probabilities and do not define a universal real/fake threshold. The
AUC measures ranking only on these three genuine and four fake clips; this small
demonstration is not a generalization benchmark.

The recorded batch-size-8 run produced **AUC 1.000** and a **+3.273773
genuine/fake margin**: every genuine query ranked above every fake query.

Source footage remains subject to its original owners' rights and terms. The
repository's MIT license does not grant redistribution rights for that footage.
