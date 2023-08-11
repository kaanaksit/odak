import torch


def cross_product(vector1, vector2):
    """
    Definition to cross product two vectors and return the resultant vector. Used method described under: http://en.wikipedia.org/wiki/Cross_product

    Parameters
    ----------
    vector1      : torch.tensor
                   A vector/ray.
    vector2      : torch.tensor
                   A vector/ray.

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray.
    """
    angle = torch.cross(vector1[1].T, vector2[1].T)
    angle = torch.tensor(angle)
    ray = torch.tensor([vector1[0], angle], dtype=torch.float32)
    return ray


def same_side(p1, p2, a, b):
    """
    Definition to figure which side a point is on with respect to a line and a point. See http://www.blackpawn.com/texts/pointinpoly/ for more. If p1 and p2 are on the sameside, this definition returns True.

    Parameters
    ----------
    p1          : list
                  Point(s) to check.
    p2          : list
                  This is the point check against.
    a           : list
                  First point that forms the line.
    b           : list
                  Second point that forms the line.
    """
    ba = torch.subtract(b, a)
    p1a = torch.subtract(p1, a)
    p2a = torch.subtract(p2, a)
    cp1 = torch.cross(ba, p1a)
    cp2 = torch.cross(ba, p2a)
    test = torch.dot(cp1, cp2)
    if len(p1.shape) > 1:
        return test >= 0
    if test >= 0:
        return True
    return False


def distance_between_two_points(point1, point2):
    """
    Definition to calculate distance between two given points.

    Parameters
    ----------
    point1      : torch.Tensor
                  First point in X,Y,Z.
    point2      : torch.Tensor
                  Second point in X,Y,Z.

    Returns
    ----------
    distance    : torch.Tensor
                  Distance in between given two points.
    """
    point1 = torch.tensor(point1) if not isinstance(point1, torch.Tensor) else point1
    point2 = torch.tensor(point2) if not isinstance(point2, torch.Tensor) else point2
    
    if len(point1.shape) == 1 and len(point2.shape) == 1:
        distance = torch.sqrt(torch.sum((point1 - point2) ** 2))
    elif len(point1.shape) == 2 or len(point2.shape) == 2:
        distance = torch.sqrt(torch.sum((point1 - point2) ** 2, dim=-1))
    
    return distance



