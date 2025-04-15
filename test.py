from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
import torch



args = parse_args()
cfg = load_config(args, "/home/awd8324/onkar/SlowFast/configs/Kinetics/MVITv2_L_40x3_test.yaml")

model = build_model(cfg)

model.head.projection = torch.nn.Linear(2048, 2)
print(model)

# check = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH)
# model.load_state_dict(check['model_state'], strict=True)
