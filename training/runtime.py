def local_yolo():
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    # The package's config directory is isolated in training.__init__.
    keys=('sync','hub','clearml','comet','dvc','mlflow','neptune','raytune','wandb','tensorboard')
    SETTINGS.update({key:False for key in keys})
    return YOLO
