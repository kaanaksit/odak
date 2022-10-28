import sys
from odak.learn.tools import load_image, save_image

def test():
    image = load_image("sample.png")
    save_image("sample_out.png", image, bit=16, cmin=0., cmax=65535.)
    assert True == True

if __name__ == '__main__':
    sys.exit(test())