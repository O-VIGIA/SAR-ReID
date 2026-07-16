# AG-VPReID preparation

The dataset is not redistributed with this repository. Request or download it
from the [official AG-VPReID repository](https://github.com/agvpreid25/AG-VPReID)
and follow its access terms.

The official camera mapping is:

- `C0`, `C1`: CCTV (ground view)
- `C2`, `C3`: wearable cameras (ground view)
- `C4`, `C5`: UAV cameras (aerial view)

STAR-CVI uses OpenGait pickle sequences arranged as:

```text
<dataset_root>/
└── <person_id>/
    └── <camera_id>/
        └── <tracklet_id>/
            └── rgb.pkl       # uint8 [T, 3, 192, 96]
```

From the STAR-CVI repository root, convert raw frame directories with:

```bash
python scripts/prepare_ag_vpreid.py \
  --train-root /path/to/AG-VPReID/train \
  --test-root /path/to/AG-VPReID/test \
  --output-root ./data/AG-VPReID_OpenGait_PKL_192_96 \
  --partition-out ./datasets/AG-VPReID/AG_VPReID.json
```

The source roots must follow `<root>/<person_id>/<tracklet>/<frames>`, and frame
names must contain their `C0`-`C5` camera token. If the test release is split
across several directories, repeat `--test-root` for each non-overlapping root.

After conversion:

1. Check the identity count printed by the script.
2. Set `model_cfg.SeparateBNNecks.class_num` in the experiment config to the
   number of training identities.
3. Set `data_cfg.dataset_root` to the generated dataset root.
4. Keep the generated `AG_VPReID.json` private if the dataset terms prohibit
   redistribution of split metadata.

