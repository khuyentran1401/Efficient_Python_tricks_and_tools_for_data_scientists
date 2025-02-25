<<<<<<< HEAD
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
||||||| dfbda73d
=======
from utils import get_mean  

get_mean(1, 3)
>>>>>>> cc473f8bc8c4403edc9964604fc4df46e7b8eba2
