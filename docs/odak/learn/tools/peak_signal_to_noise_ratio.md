# odak.save_image

`save_image(fn,img,cmin=0,cmax=255)`

Definition to save a Numpy array as an image.
 
**Parameters:**
                       
    fn           : str
                   Filename.
    img          : ndarray
                   A numpy array with NxMx3 or NxMx1 shapes.
    cmin         : int
                   Minimum value that will be interpreted as 0 level in the final image.
    cmax         : int
                   Maximum value that will be interpreted as 255 level in the final image.

**Returns**

    bool         :  bool
                    True if successful.

## See also

* [`General toolkit`](../../../toolkit.md)
