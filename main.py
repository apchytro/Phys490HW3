from data_gen import Data
import numpy as np
import torch, argparse
import matplotlib.pyplot as plt
import os

# PHYS 490 HW3 
# Anthony Chytros
# 20624286

# Generative model to determine Ising model couplers 

# Function to determine the hamiltonian for a spin configuration
# array[float] array[int] int --> float
def hamiltonian(coupling,sigma,N):
        sum = 0
        for i in range(len(sigma)):
            sum -= coupling[i] * sigma[i] * sigma [(i+1)%N]
        return sum

# Function to determine the partition function
# float array[array[int]] array[float] int --> float
def partition(beta,sigmas,coupling, N):
        part = 0.0
        for i in sigmas:
            part += np.exp(-1 * beta * hamiltonian(coupling,i,N))
        return part
    
# Function to determine the probability of all spin configurations
# float array[array[int]] array[float] int --> array[float]
def prob(beta,sigmas,coupling, N):
        temp = []
        part = partition(beta,sigmas,coupling, N)
        for i in sigmas:
            temp.append(np.exp(-1 * beta * hamiltonian(coupling,i,N))/part)
        return np.array(temp)

# Function to determine the expectation value of spin_i spin_j of a specified probability distribution
# int int array[float] array[array[int]] --> float
def expectation(i,j,prob,sigma):
    exp = 0
    for k in range(len(sigma)):
        exp += prob[k] * sigma[k][i] * sigma[k][j]
    return exp
  
    

if __name__ == '__main__':
    
    # Input parser arguments
    parser = argparse.ArgumentParser(description='Generative Ising Model')
    parser.add_argument('-v', type=int, default=1, metavar='N', help='verbosity (default: 1)')
    parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='beta (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='epochs', help='number of epochs (default: 2e2)')
    parser.add_argument('--lr', type=float, default=0.3, metavar='lr', help='learning rate (default: 3e-1)')
    parser.add_argument('--data', default='data\in.txt', metavar='in.txt', help='data directory (default: data\in.txt)')
    parser.add_argument('--res', default ='results', metavar='results', help='path of results (default: results)')
    args = parser.parse_args()
    
    # Load data from specified path
    data = Data(datafile = args.data)
    # Define loss function
    loss= torch.nn.KLDivLoss(reduction='batchmean')
    # Initialize beta, epochs and learning rate
    beta = args.beta
    num_epochs= args.epochs
    lr=args.lr
    # Get number of spins per configuration
    N = np.size(data.x,1)
    # Initialize a 'guess' of coupling values
    coupling = np.ones(N, dtype = float)
    # Convert probability distribution from dataset to a torch array
    targets = torch.from_numpy(data.y)
    # Initialize array for storing training losses
    train_loss= []
    
    # Itereate epoch times updating coupling weights and storing losses
    for epoch in range(1, num_epochs + 1):
        # Calculate model probability distribution w current weights
        lambda_prob = prob(beta,data.x,coupling,N)
        # Convert to torch and calculate loss
        outputs = torch.from_numpy(np.log(lambda_prob))
        train_loss.append(loss(outputs,targets).item())
        # Update weights using KLDiv update rule 
        temp_coupling = []
        for lam in range(len(coupling)):
            temp_coupling.append((beta * (expectation(lam,((lam+1)%N),data.y,data.x) - expectation(lam,((lam+1)%N),lambda_prob,data.x))))
        coupling += lr * (np.array(temp_coupling))
    
    # If verbosity is large plot losses and save figure
    if args.v >= 2:
        plt.plot (np.arange(1, num_epochs + 1), train_loss)
        plt.title ("KLDiv Losses versus Number of Epochs")
        plt.xlabel ("Number of Epochs")
        plt.ylabel ("Loss")
        plt.savefig(os.path.join(args.res, 'plot.pdf'))
        plt.close()
    
    # Convert final values to dictionary of +1 / -1 and print
    final = {}
    for i in range (len(coupling)):
        final[(i,(i+1)%N)] = int(round(coupling[i]))
    print (final)
    
    f = open(os.path.join(args.res, "results.txt"),"w")
    f.write( str(final) )
    f.close()
    
