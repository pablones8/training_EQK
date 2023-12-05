import numpy as np
from pennylane import numpy as np

def sinus_data(samples):
    """
    Generates a dataset of points with 1/0 labels
    depending on whether they are above or below the sine function
    Args:
        samples (int): number of samples to generate

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        f= 0.8*np.sin(-2*np.pi*x[0]/2)
        point = x[1]
        if f < point:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


def sinus_data_dif(samples):
    """
    Generates a dataset of points with 1/0 labels
    depending on whether they are above or below the sine function
    Args:
        samples (int): number of samples to generate

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        #f= np.sin(2*np.pi*x[0]/2+np.pi/2)
        f= np.sin(-2*np.pi*x[0]/1)
        point = x[1]
        if f < point:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


def corners(samples):
    Xvals, yvals = [], []

    tl=[-1.0, 1.0]
    tr=[1.0, 1.0]
    bl=[-1.0, -1.0]
    br=[1.0, -1.0]

    radius = 0.75

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0

        d_tl = np.linalg.norm(x-tl)
        d_tr = np.linalg.norm(x-tr)
        d_bl = np.linalg.norm(x-bl)
        d_br = np.linalg.norm(x-br)

        if d_tl<radius or d_tr<radius or d_bl<radius or d_br<radius:
            y=1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)



def circles(samples, center=[0.0, 0.0], radius1=np.sqrt(2 / np.pi), radius2=0.5*np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius1 and np.linalg.norm(x - center) > radius2:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)



def spiral(N):
    N=N//2
    theta = np.sqrt(np.random.rand(N))*2*np.pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = (data_a + np.random.randn(N,2))/20

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = (data_b + np.random.randn(N,2))/20
    xvals = np.append(x_a, x_b, axis=0)
    yvals = np.concatenate([np.ones(N),np.zeros(N)],axis=0)

    data = list(zip(np.array(xvals, requires_grad=False), np.array(yvals, requires_grad=False)))
    np.random.shuffle(data)

    equis, yes = zip(*data)

    equis, yes = np.array(equis, requires_grad=False), np.array(yes, requires_grad=False)

    return equis, yes