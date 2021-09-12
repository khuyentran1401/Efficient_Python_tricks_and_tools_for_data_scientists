import hydra 

@hydra.main(config_name='config.yaml')
def main(config):
    print(f'Process {config.data}')
    print(f'Drop features: {config.variables.drop_features}')

if __name__ == '__main__':
    main()