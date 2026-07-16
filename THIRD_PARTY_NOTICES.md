# Third-party notices

## OpenAI CLIP

The following files are derived from or distributed with
[OpenAI CLIP](https://github.com/openai/CLIP):

- `opengait/modeling/model_clip/clip/clip.py`
- `opengait/modeling/model_clip/clip/model.py`
- `opengait/modeling/model_clip/clip/simple_tokenizer.py`
- `opengait/modeling/model_clip/clip/bpe_simple_vocab_16e6.txt.gz`

STAR-CVI modifies the visual encoder interface, input grid/stride handling,
position-embedding resize, and intermediate token outputs. OpenAI CLIP is
licensed under the MIT License. A verbatim copy is available at
`third_party/openai_clip/LICENSE`.

## OpenGait

STAR-CVI is an additive extension for
[OpenGait](https://github.com/ShiqiYu/OpenGait). The full OpenGait source is not
vendored here. Users must obtain it separately and comply with its upstream
terms, including the academic-use notice in its repository.

## AG-VPReID

No AG-VPReID data are distributed here. Users must obtain the dataset from the
[official repository](https://github.com/agvpreid25/AG-VPReID) and comply with
its access and usage terms.

