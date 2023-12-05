# Constructing Embedding Quantum Kernels from Quantum Neural Network training

In this repository, we demonstrate how to train an Embedding Quantum Kernel (EQK) by leveraging the training of a data re-uploading Quantum Neural Network (QNN). Specifically, in the `main.ipynb` notebook, we employ the architecture denoted as $1$-to-$n$. Here, we utilize the training of a single-qubit data re-uploading QNN with $L=7$ layers to construct an $n$-qubit EQK with entanglement between layers. The notebook includes simulations for the case when $n=3$. The explicit construction of the EQK is illustrated in the figure below:



Surprisingly, a single-qubit QNN is capable of selecting optimal parameters to construct a larger EQK. The subsequent figure displays the results of adding qubits with the same architecture as in the previous figure, with entanglement represented by a cascade of CNOTs.



The results reveal that adding qubits can enhance accuracy, even when the training remains consistent across all cases.
