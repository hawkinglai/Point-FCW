from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def adjust_values(tensor):
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Shift and scale the tensor values so that 0.9 maps to 0
    shifted_tensor = (tensor - 0.9) * 10

    # Apply the sigmoid function
    adjusted_tensor = sigmoid(shifted_tensor)

    return adjusted_tensor

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item()
    acc = 100 * acc / target.shape[0]
    return acc

def classification(feature_bank, label_bank, test_feature_bank, test_label_bank, plot=False):
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_gamma = 0, 0
    Sim = test_feature_bank.cuda().float() @ feature_bank.cuda().permute(1, 0).float()
    for gamma in tqdm(gamma_list, desc="Searching best gamma"):
        logits = (-gamma * (1 - Sim)).exp() @ label_bank
        acc = cls_acc(logits, test_label_bank)

        if acc > best_acc:
            best_acc, best_gamma = acc, gamma

    print(f"TDA's classification accuracy: {best_acc:.2f}.")
    print(f"TDA's best gamma: {best_gamma:.2f}.")

    if plot == True:
        tensor_np = Sim.cpu().numpy()
        adjusted_tensor = adjust_values(tensor_np)
        # print(adjusted_tensor)
        plt.imshow(adjusted_tensor, cmap='hot', interpolation='nearest')
        plt.savefig('/pointtda/plot2.pdf', bbox_inches='tight')
