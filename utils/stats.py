from include.imports import *

def plot_convergence(history, model_name, angulo, i, mensagem = ""):
    
        os.makedirs(f"history/{model_name}", exist_ok=True)
    
        # Gráfico de perda de treinamento
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title(f'Training Loss Convergence for {model_name} - {angulo} - Run {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"history/{model_name}/{mensagem}_{angulo}_{i}_training_loss_convergence.png")
        plt.close()

        # Gráfico de perda de validação
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Validation Loss Convergence for {model_name} - {angulo} - Run {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"history/{model_name}/{mensagem}_{angulo}_{i}_validation_loss_convergence.png")
        plt.close()

        
        




    


def precision_score_(y_true, y_pred, eps=1e-7):
    intersect = np.sum(y_pred & y_true)
    total_pred = np.sum(y_pred)
    return round(intersect / (total_pred + eps), 3)

def recall_score_(y_true, y_pred, eps=1e-7):
    intersect = np.sum(y_pred & y_true)
    total_true = np.sum(y_true)
    return round(intersect / (total_true + eps), 3)

def accuracy_score_(y_true, y_pred):
    return round(np.mean(y_true == y_pred), 3)

def dice_coef_(y_true, y_pred, eps=1e-7):
    intersect = np.sum(y_pred & y_true)
    total = np.sum(y_pred) + np.sum(y_true)
    return round((2 * intersect) / (total + eps), 3)

def iou_(y_true, y_pred, eps=1e-7):
    intersect = np.sum(y_pred & y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersect
    return round(intersect / (union + eps), 3)
