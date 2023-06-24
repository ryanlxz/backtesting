import yaml
import pathlib

path = pathlib.Path(__file__).parent / "conf.yaml"
with path.open(mode="rb") as yamlfile:
    backtest_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
