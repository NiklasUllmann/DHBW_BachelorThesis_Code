# Function to simulate XOR Gate
def XOR(A, B):
    return A ^ B

# Function to simulate NOT Gate


def NOT(A):
    return ~A+2

# Function to simulate XNOR Gate


def XNOR(A, B):
    return NOT(XOR(A, B))
