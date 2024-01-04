import wandb

sweep_configuration = {
    'command': '\n - ${env}\n - python3\n - tune_hyperparameters.py\n - ${args}',
    'early_terminate':
    {
        'type': 'hyperband',
        'min_iter': 20
    },
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        'lr10': {'min': -4.0, 'max': -1.0},
        'mesh_n_max': {'min': 500, 'max': 3000},
        'mesh_radius': {'min': 100.0, 'max': 500.0},
        'model_width': {'min': 10, 'max': 64},
        'model_kernel_width': {'min': 10, 'max': 64},
        'model_depth': {'min': 1, 'max': 8}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project= "wind_to_waves")