import os
import config

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def log_hyperparams(exp_name, include_dice=False):

    if include_dice is True:
        config.EMBEDDING_LOSS_TYPE = "dice_" + config.EMBEDDING_LOSS_TYPE


    hyperparams_log_file = os.path.join("logs", exp_name + "_hyperparams.txt")
    hp_fh = open(hyperparams_log_file, 'w+')

    hp_fh.write(f"Batch Size: {config.BATCH_SIZE}\n")
    hp_fh.write(f"Epochs: {config.NUM_EPOCHS}\n")
    hp_fh.write(f"Generator Pre-Training Epochs: {config.GR_PRE_TRAINING_EPOCHS}\n")
    hp_fh.write(f"Generator Learning Rate: {config.GR_LEARNING_RATE}\n")
    hp_fh.write(f"Discriminator Learning Rate: {config.DR_LEARNING_RATE}\n")
    hp_fh.write(f"Embedding Loss Type: {config.EMBEDDING_LOSS_TYPE}\n")

    hp_fh.close()
